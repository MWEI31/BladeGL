"""CBAM fusion head for full-image and local features.

Channel and spatial attention follow the reference implementation:
https://github.com/Jongchan/attention-module/MODELS/cbam.py
"""
import torch
from torch import nn
from torch.nn import functional as F

from .backbone import Head as SharedHead
from .resize import stable_resize


class CBAM(nn.Module):
    def __init__(self, channels=192, reduction_ratio=16, spatial_kernel=7):
        super().__init__()
        if (channels, reduction_ratio, spatial_kernel) != (192, 16, 7):
            raise ValueError('E01 locks CBAM to 192 channels, ratio 16, kernel 7')
        self.channel_mlp = nn.Sequential(
            nn.Linear(channels, channels // reduction_ratio), nn.ReLU(),
            nn.Linear(channels // reduction_ratio, channels))
        self.spatial_conv = nn.Conv2d(2, 1, spatial_kernel,
                                      padding=spatial_kernel // 2, bias=False)
        self.spatial_bn = nn.BatchNorm2d(1, eps=1e-5, momentum=.01, affine=True)

    def forward(self, features):
        avg = F.avg_pool2d(features, features.shape[-2:]).flatten(1)
        maximum = F.max_pool2d(features, features.shape[-2:]).flatten(1)
        channel_weight = (self.channel_mlp(avg) + self.channel_mlp(maximum)).sigmoid()
        weighted = features * channel_weight[:, :, None, None]
        spatial_input = torch.cat((weighted.max(dim=1, keepdim=True).values,
                                   weighted.mean(dim=1, keepdim=True)), dim=1)
        spatial_weight = self.spatial_bn(self.spatial_conv(spatial_input)).sigmoid()
        return weighted * spatial_weight


class Head(SharedHead):
    def __init__(self):
        # Initialize the shared head before the CBAM module.
        super().__init__('concat')
        self.study_arm = 'E01_CBAM'
        self.cbam = CBAM()

    def forward(self, features):
        global_features = stable_resize(self.project(features['global_features']), (72, 72))
        local_features = self.stitch(features['tiles'])
        fused = self.fuse(self.cbam(torch.cat((global_features, local_features), dim=1)))
        z = self.decode1(stable_resize(fused, (144, 144)))
        z = self.segment(self.decode2(stable_resize(z, (288, 288))))[:, 0]
        q = z.float().sigmoid().flatten(1).topk(830, dim=1).values.mean(1, keepdim=True)
        score = self.detect(torch.cat((self.cls_norm(features['cls']), q), dim=1))[:, 0]
        return z, score


def make_head(arm):
    if arm != 'E01_CBAM':
        raise ValueError(f'E01 cannot instantiate another experiment: {arm}')
    return Head()


def required_views(arm):
    if arm != 'E01_CBAM':
        raise ValueError(arm)
    return False, True, True


def loss_value(arm, z, score, target, valid, labels):
    if arm != 'E01_CBAM':
        raise ValueError(arm)
    bce = F.binary_cross_entropy_with_logits(
        z.float(), target.float(), pos_weight=z.new_tensor(8.).float(), reduction='none')
    bce = (bce * valid).sum() / valid.sum().clamp_min(1)
    prob = z.float().sigmoid() * valid
    abnormal = labels > 0
    dice = 1 - (2 * (prob * target).sum((1, 2)) + 1) / (
        prob.sum((1, 2)) + target.sum((1, 2)) + 1)
    dl = dice[abnormal].mean() if abnormal.any() else z.sum() * 0
    image = F.binary_cross_entropy_with_logits(score.float(), labels)
    return bce + .3 * dl + .2 * image
