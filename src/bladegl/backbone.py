"""Shared DINOv2 encoder and detection/localization heads."""
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
        # Encoder inputs are square; non-square images require letterboxing.
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
        assert arm in ('global','direct_high','concat')
        self.arm=arm
        self.project=convnorm(1152,96,1)
        self.decode1=convnorm(96,64)
        self.decode2=convnorm(64,32)
        self.segment=nn.Conv2d(32,1,1)
        self.cls_norm=nn.LayerNorm(384)
        self.detect=nn.Sequential(nn.Linear(385,64),nn.GELU(),nn.Linear(64,1))
        if arm=='concat': self.fuse=convnorm(192,96)
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

    def forward(self, features):
        if self.arm=='direct_high':
            fused=self.project(features['high_features'])
        else:
            g=stable_resize(self.project(features['global_features']),(72,72))
            if self.arm=='global':fused=g
            else:
                l=self.stitch(features['tiles'])
                fused=self.fuse(torch.cat([g,l],1))
        z=self.decode1(stable_resize(fused,(144,144)))
        z=self.segment(self.decode2(stable_resize(z,(288,288))))[:,0]
        q=z.float().sigmoid().flatten(1).topk(830,dim=1).values.mean(1,keepdim=True)
        normalized_cls=self.cls_norm(features['cls'])
        score=self.detect(torch.cat([normalized_cls,q],dim=1))[:,0]
        return z,score
