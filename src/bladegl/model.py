"""Final fusion candidates with explicit FP32 attention and shared spatial protocol.

Cosine logits adapt Swin V2; residual depthwise projections adapt CvT's
convolutional attention idea. These operators are task adaptations.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from .backbone import Head as OriginalHead
from .resize import stable_resize

MAIN_ARMS = ('baseline', 'shift_dot', 'shift_cos', 'shift_conv', 'shift_cos_conv')
ABLATIONS = ('no_position', 'no_shift', 'self', 'two_regular')


class LocalExchange(nn.Module):
    """Local queries contextual keys; regular and shifted views share weights.

    two_regular changes depth as well as geometry; it is a comparable-work
    alternative, not a pure shift toggle. self changes K/V to local features
    while the final global fusion pathway stays intact.
    """
    def __init__(self, cosine=False, convolution=False, position=True,
                 shifted=True, self_attention=False, two_regular=False,
                 window=6, dim=24, heads=2):
        super().__init__()
        assert 72 % window == 0 and dim % heads == 0
        self.window, self.dim, self.heads = window, dim, heads
        self.shift = window // 2
        self.cosine, self.convolution = cosine, convolution
        self.shifted, self.self_attention = shifted, self_attention
        self.two_regular = two_regular
        self.norm_g, self.norm_l = nn.LayerNorm(96), nn.LayerNorm(96)
        # Keep historical common attention initialization before removing
        # the unused direction; all candidate differences are added afterward.
        qkv_g, qkv_l = nn.Linear(96, 3*dim), nn.Linear(96, 3*dim)
        unused_out_g = nn.Linear(dim, 96)
        self.out_l = nn.Linear(dim, 96)
        self.query_l = nn.Linear(96, dim)
        self.key_value_g = nn.Linear(96, 2*dim)
        with torch.no_grad():
            self.query_l.weight.copy_(qkv_l.weight[:dim])
            self.query_l.bias.copy_(qkv_l.bias[:dim])
            self.key_value_g.weight.copy_(qkv_g.weight[dim:])
            self.key_value_g.bias.copy_(qkv_g.bias[dim:])
        del qkv_g, qkv_l, unused_out_g
        self.scale_l = nn.Parameter(torch.zeros(()))
        if cosine:
            self.log_temperature = nn.Parameter(torch.full((heads,1,1), math.log(10.)))
        else:
            self.register_parameter('log_temperature', None)
        if convolution:
            self.depthwise_g = nn.Conv2d(96,96,3,padding=1,groups=96,bias=False)
            self.depthwise_l = nn.Conv2d(96,96,3,padding=1,groups=96,bias=False)
            nn.init.zeros_(self.depthwise_g.weight)
            nn.init.zeros_(self.depthwise_l.weight)
        s = window
        yy,xx = torch.meshgrid(torch.arange(s),torch.arange(s),indexing='ij')
        coordinates = torch.stack([yy,xx]).flatten(1)
        offset = coordinates[:,:,None]-coordinates[:,None,:]+s-1
        self.register_buffer('relative_index',offset[0]*(2*s-1)+offset[1],persistent=False)
        if position:
            self.relative_bias = nn.Parameter(torch.zeros((2*s-1)**2,heads))
        else:
            self.register_parameter('relative_bias',None)
        yy,xx = torch.meshgrid(torch.arange(72),torch.arange(72),indexing='ij')
        coordinates = torch.stack([yy,xx]).unsqueeze(0).float()
        tokens = self.partition(torch.roll(coordinates,(-self.shift,)*2,(-2,-1)))
        blocked = (tokens[:,:,None,:]-tokens[:,None,:,:]).abs().amax(-1) >= s
        self.register_buffer('shift_blocked',blocked,persistent=False)

    def partition(self,x):
        b,c,h,w = x.shape
        s = self.window
        assert h % s == 0 and w % s == 0
        return x.reshape(b,c,h//s,s,w//s,s).permute(0,2,4,3,5,1).reshape(-1,s*s,c)

    def restore(self,tokens,shape):
        b,c,h,w = shape
        s = self.window
        return tokens.reshape(b,h//s,w//s,s,s,c).permute(0,5,1,3,2,4).reshape(shape)

    def attention(self,q,k,v,shifted=False):
        # .float() alone does not prevent autocast from lowering a matmul.
        with torch.autocast(device_type=q.device.type,enabled=False):
            q32,k32,v32 = q.float(),k.float(),v.float()
            if self.cosine:
                q32 = F.normalize(q32,dim=-1,eps=1e-6)
                k32 = F.normalize(k32,dim=-1,eps=1e-6)
                temperature = self.log_temperature.float().clamp(max=math.log(100.)).exp()
                logits = (q32 @ k32.transpose(-1,-2))*temperature[None]
            else:
                logits = (q32 @ k32.transpose(-1,-2))/math.sqrt(q32.shape[-1])
            if self.relative_bias is not None:
                bias = self.relative_bias.float()[self.relative_index].permute(2,0,1)
                logits = logits+bias[None]
            if shifted:
                nw = self.shift_blocked.shape[0]
                logits = logits.reshape(-1,nw,self.heads,self.window**2,self.window**2)
                logits = logits.masked_fill(self.shift_blocked[None,:,None],float('-inf'))
                logits = logits.flatten(0,1)
            weights = logits.softmax(dim=-1)
            # Identical FP32 value aggregation for every candidate.
            result = (weights @ v32).to(v.dtype)
        return result.transpose(1,2).reshape(q.shape[0],q.shape[2],self.dim)

    def delta(self,g,l,shifted=False):
        key_source = l if self.self_attention else g
        if self.convolution:
            # Convolve before rolling, avoiding artificial cyclic boundaries.
            key_source = key_source+self.depthwise_g(key_source)
            query_source = l+self.depthwise_l(l)
        else:
            query_source = l
        if shifted:
            key_source = torch.roll(key_source,(-self.shift,)*2,(-2,-1))
            query_source = torch.roll(query_source,(-self.shift,)*2,(-2,-1))
        gt = self.norm_g(self.partition(key_source))
        lt = self.norm_l(self.partition(query_source))
        n,t,_ = lt.shape
        q = self.query_l(lt).reshape(n,t,self.heads,self.dim//self.heads).transpose(1,2)
        k,v = self.key_value_g(gt).reshape(n,t,2,self.heads,self.dim//self.heads).permute(2,0,3,1,4).unbind(0)
        dl = self.restore(self.out_l(self.attention(q,k,v,shifted)),l.shape)
        return torch.roll(dl,(self.shift,)*2,(-2,-1)) if shifted else dl

    def forward(self,g,l):
        scale = self.scale_l.tanh()
        dl = self.delta(g,l)
        if self.two_regular:
            # Two half-strength sequential updates keep total residual strength
            # comparable while creating a meaningful same-parameter depth control.
            middle = l+.5*scale*dl
            return g,middle+.5*scale*self.delta(g,middle)
        if self.shifted:
            dl = .5*(dl+self.delta(g,l,shifted=True))
        return g,l+scale*dl


class Head(OriginalHead):
    def __init__(self,study_arm,window=6,region_input='normal',image_loss=True,
                 overlap='average'):
        # Common projection, decoder and classifier get paired initialization.
        super().__init__('concat')
        pieces = study_arm.split('__')
        assert len(pieces) <= 2, 'Use exactly one optional ablation suffix'
        main = pieces[0]
        ablation = pieces[1] if len(pieces)==2 else None
        assert main in MAIN_ARMS,study_arm
        assert ablation is None or ablation in ABLATIONS,study_arm
        assert main!='baseline' or ablation is None,'Baseline has no exchange to ablate'
        self.study_arm = study_arm
        if region_input not in ('normal','zero','detach'):
            raise ValueError(region_input)
        if overlap not in ('average','fixed_priority'):
            raise ValueError(overlap)
        self.region_input=region_input
        self.image_loss=image_loss
        self.overlap=overlap
        if main!='baseline':
            self.exchange = LocalExchange(
                cosine=main in ('shift_cos','shift_cos_conv'),
                convolution=main in ('shift_conv','shift_cos_conv'),
                position=ablation!='no_position',
                shifted=ablation not in ('no_shift','two_regular'),
                self_attention=ablation=='self',
                two_regular=ablation=='two_regular',window=window)

    def stitch(self,tiles):
        if self.overlap=='average':
            return super().stitch(tiles)
        b=tiles.shape[0]
        p=self.project(tiles.reshape(b*4,1152,37,37)).reshape(b,4,96,37,37)
        full=p.new_zeros(b,96,72,72)
        filled=torch.zeros((72,72),dtype=torch.bool,device=p.device)
        for j,(y,x) in enumerate(((0,0),(0,35),(35,0),(35,35))):
            free=~filled[y:y+37,x:x+37]
            region=full[:,:,y:y+37,x:x+37]
            full[:,:,y:y+37,x:x+37]=torch.where(free[None,None],p[:,j],region)
            filled[y:y+37,x:x+37]=True
        return full

    def fuse_views(self,g,l):
        if hasattr(self,'exchange'):
            g,l = self.exchange(g,l)
        return self.fuse(torch.cat([g,l],dim=1))

    def forward(self,features):
        g = stable_resize(self.project(features['global_features']),(72,72))
        l = self.stitch(features['tiles'])
        fused = self.fuse_views(g,l)
        z = self.decode1(stable_resize(fused,(144,144)))
        z = self.segment(self.decode2(stable_resize(z,(288,288))))[:,0]
        q = z.float().sigmoid().flatten(1).topk(830,dim=1).values.mean(1,keepdim=True)
        if self.region_input=='zero':q=torch.zeros_like(q)
        elif self.region_input=='detach':q=q.detach()
        score = self.detect(torch.cat([self.cls_norm(features['cls']),q],dim=1))[:,0]
        return z,score

