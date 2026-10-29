"""Fixed align_corners=False bilinear geometry with FP32 separable matmuls.

Avoids atomic accumulation in CUDA interpolation backward.
"""
from functools import lru_cache
import torch


@lru_cache(maxsize=24)
def _weights(n_in,n_out,device):
    x=(torch.arange(n_out,dtype=torch.float64)+.5)*(n_in/n_out)-.5
    x=x.clamp(0,n_in-1)
    lo=x.floor().long();hi=(lo+1).clamp(max=n_in-1);w=x-lo
    m=torch.zeros(n_out,n_in,dtype=torch.float64)
    rows=torch.arange(n_out)
    m[rows,lo]+=1-w;m[rows,hi]+=w
    return m.float().to(torch.device(device))


def stable_resize(x,size):
    h,w=(size,size) if isinstance(size,int) else size
    if x.shape[-2:]==(h,w):return x
    mh=_weights(x.shape[-2],h,str(x.device));mw=_weights(x.shape[-1],w,str(x.device))
    with torch.autocast(device_type=x.device.type,enabled=False):
        result=torch.matmul(torch.matmul(mh,x.float()),mw.t())
    return result.to(x.dtype)
