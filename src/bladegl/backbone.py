"""Complete shared-backbone global/local detection and segmentation model."""
import os
import time
os.environ.setdefault('XFORMERS_DISABLED', '1')
import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .resize import stable_resize


class Encoder(nn.Module):
    def __init__(self, root, cfg):
        super().__init__()
        self.mixed_precision=bool(cfg.get('mixed_precision',True))
        self.net = torch.hub.load(str(root/cfg['dino_repo']), 'dinov2_vits14', source='local', pretrained=False)
        self.net.load_state_dict(torch.load(root/cfg['dino_weights'], map_location='cpu', weights_only=True), strict=True)
        self.net.eval().requires_grad_(False)
        self.register_buffer('mean', torch.tensor([.485,.456,.406])[None,:,None,None])
        self.register_buffer('std', torch.tensor([.229,.224,.225])[None,:,None,None])

    def forward(self, x):
        layers = self.net.get_intermediate_layers((x-self.mean)/self.std, n=[5,8,11], return_class_token=True)
        side=x.shape[-1]//14
        feat=torch.cat([F.normalize(p.float(),dim=-1) for p,c in layers],dim=-1)
        feat=feat.transpose(1,2).reshape(len(x),1152,side,side)
        cls=F.normalize(layers[-1][1].float(),dim=-1)
        return feat,cls

    @torch.inference_mode()
    def extract(self, rgb, include_direct=True, include_tiles=True, include_global=True,
                stage_ms=None):
        # The current released split is entirely square. Fail explicitly for new geometry.
        assert rgb.ndim==3 and rgb.shape[0]==rgb.shape[1], 'Non-square input requires the documented letterbox geometry adapter.'
        device=self.mean.device
        def measured(name, fn):
            if stage_ms is None:
                return fn()
            if device.type == 'cuda': torch.cuda.synchronize(device)
            start = time.perf_counter_ns()
            value = fn()
            if device.type == 'cuda': torch.cuda.synchronize(device)
            stage_ms[name] = stage_ms.get(name, 0.) + (time.perf_counter_ns()-start)/1e6
            return value
        low=measured('view_preparation', lambda: cv2.resize(
            rgb,(518,518),interpolation=cv2.INTER_AREA)) if include_global else None
        high=measured('view_preparation', lambda: cv2.resize(
            rgb,(1008,1008),interpolation=cv2.INTER_AREA)) if include_direct or include_tiles else None
        def tensor(a):
            host = measured('view_preparation', lambda: torch.from_numpy(
                np.ascontiguousarray(a.transpose(2,0,1))))
            return measured('host_to_device', lambda: host.to(device).float()[None]/255)
        result={}
        with torch.autocast(device_type=device.type,dtype=torch.float16,
                            enabled=self.mixed_precision and device.type=='cuda'):
            if include_global:
                global_tensor = tensor(low)
                g,c=measured('encoder', lambda: self(global_tensor));result.update(global_features=g,cls=c)
            if include_tiles:
                ts=[]
                for y,x in ((0,0),(0,490),(490,0),(490,490)):
                    tile_tensor = tensor(high[y:y+518,x:x+518])
                    t,tc=measured('encoder', lambda: self(tile_tensor));ts.append(t)
                    if 'tile_cls' not in result: result['tile_cls']=[]
                    result['tile_cls'].append(tc)
                result['tiles']=torch.stack(ts,dim=1)
                result['tile_cls']=torch.stack(result['tile_cls'],dim=1)
            if include_direct:
                high_tensor = tensor(high)
                h,hc=measured('encoder', lambda: self(high_tensor));result.update(high_features=h,high_cls=hc)
        return {k:v.half() for k,v in result.items()}


def convnorm(a,b,k=3):
    return nn.Sequential(nn.Conv2d(a,b,k,padding=k//2),nn.GroupNorm(8,b),nn.GELU())


class Head(nn.Module):
    def __init__(self, arm):
        super().__init__()
        assert arm in ('global','direct_high','concat','residual')
        self.arm=arm
        self.project=convnorm(1152,96,1)
        self.decode1=convnorm(96,64)
        self.decode2=convnorm(64,32)
        self.segment=nn.Conv2d(32,1,1)
        self.cls_norm=nn.LayerNorm(384)
        self.detect=nn.Sequential(nn.Linear(385,64),nn.GELU(),nn.Linear(64,1))
        if arm=='concat': self.fuse=convnorm(192,96)
        if arm=='residual':
            self.residual=nn.Conv2d(96,96,3,padding=1)
            self.gate=nn.Sequential(nn.Conv2d(288,32,3,padding=1),nn.GELU(),nn.Conv2d(32,1,1),nn.Sigmoid())
        count=torch.zeros(1,1,72,72)
        for y,x in ((0,0),(0,35),(35,0),(35,35)):count[:,:,y:y+37,x:x+37]+=1
        self.register_buffer('tile_count',count,persistent=False)

    def stitch(self,tiles):
        b=tiles.shape[0]
        p=self.project(tiles.reshape(b*4,1152,37,37)).reshape(b,4,96,37,37)
        full=p.new_zeros(b,96,72,72)
        for j,(y,x) in enumerate(((0,0),(0,35),(35,0),(35,35))):
            full[:,:,y:y+37,x:x+37]+=p[:,j]
        return full/self.tile_count

    def forward(self, features, return_gate=False, intervention=None):
        if self.arm=='direct_high':
            fused=self.project(features['high_features']);cls=features['cls'];gate=None
        else:
            g=stable_resize(self.project(features['global_features']),(72,72))
            cls=features['cls'];gate=None
            if self.arm=='global':fused=g
            else:
                l=self.stitch(features['tiles'])
                if self.arm=='concat':fused=self.fuse(torch.cat([g,l],1))
                else:
                    gate=self.gate(torch.cat([g,l,torch.abs(l-g)],1))
                    fused=g+gate*self.residual(l-g)
                    if intervention=='local_off':fused=g
                    elif intervention=='gate_half':fused=g+.5*self.residual(l-g)
                    elif intervention=='gate_one':fused=g+self.residual(l-g)
        z=self.decode1(stable_resize(fused,(144,144)))
        z=self.segment(self.decode2(stable_resize(z,(288,288))))[:,0]
        q=z.float().sigmoid().flatten(1).topk(830,dim=1).values.mean(1,keepdim=True)
        normalized_cls=self.cls_norm(cls)
        if intervention=='map_score_off':q=torch.zeros_like(q)
        elif intervention=='cls_off':normalized_cls=torch.zeros_like(normalized_cls)
        score=self.detect(torch.cat([normalized_cls,q],dim=1))[:,0]
        if return_gate:return z,score,gate
        return z,score


class CompleteModel(nn.Module):
    def __init__(self,root,cfg,arm='residual'):
        super().__init__()
        self.encoder=Encoder(root,cfg)
        self.head=Head(arm)

    def forward_rgb(self,rgb):
        f=self.encoder.extract(rgb,include_direct=self.head.arm=='direct_high',include_tiles=self.head.arm in ('concat','residual'),include_global=self.head.arm!='direct_high')
        with torch.autocast(device_type='cuda',dtype=torch.float16):
            z,s=self.head(f)
        return z.float().sigmoid(),s.float().sigmoid()


def loss_value(z,s,target,label):
    bce=F.binary_cross_entropy_with_logits(z.float(),target,pos_weight=z.new_tensor(8.).float())
    p=z.float().sigmoid();abnormal=label>0
    dice=1-(2*(p*target).sum((1,2))+1)/(p.sum((1,2))+target.sum((1,2))+1)
    dl=dice[abnormal].mean() if abnormal.any() else z.sum()*0
    image=F.binary_cross_entropy_with_logits(s.float(),label)
    return bce+.3*dl+.2*image
