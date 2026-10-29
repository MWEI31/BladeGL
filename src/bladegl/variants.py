"""BladeGL model variants."""
import torch
from torch.nn import functional as F

from .backbone import Head as BasicHead
from .model import Head as FusionHead
from .resize import stable_resize


FUSION_ARMS = {
    'F0': 'baseline', 'F1': 'shift_dot', 'F2': 'shift_cos',
    'F3': 'shift_conv', 'F5': 'shift_cos_conv__no_position',
    'F6': 'shift_cos_conv__no_shift', 'F7': 'shift_cos_conv__self',
    'F8': 'shift_cos_conv__two_regular',
}
SUPPORTED_ARMS = frozenset(FUSION_ARMS) | {
    'E01_CBAM', 'S1', 'S2', 'S3', 'S6', 'R4', 'R5', 'R6',
    'W4', 'W6', 'W8', 'E0', 'S5', 'F4',
}


class LocalOnlyHead(BasicHead):
    """S3: local spatial evidence with the common global CLS readout."""

    def __init__(self):
        super().__init__('global')

    def forward(self, features):
        local = self.stitch(features['tiles'])
        z = self.decode1(stable_resize(local, (144, 144)))
        z = self.segment(self.decode2(stable_resize(z, (288, 288))))[:, 0]
        q = z.float().sigmoid().flatten(1).topk(830, dim=1).values.mean(1, keepdim=True)
        cls = features['cls']
        score = self.detect(torch.cat([self.cls_norm(cls), q], dim=1))[:, 0]
        return z, score


def make_head(arm: str):
    if arm not in SUPPORTED_ARMS:
        raise ValueError(f'Unknown BladeGL arm: {arm}')
    if arm == 'E01_CBAM':
        from .cbam import Head
        return Head()
    if arm == 'S1':
        return BasicHead('global')
    if arm == 'S2':
        return BasicHead('direct_high')
    if arm == 'S3':
        return LocalOnlyHead()
    if arm == 'S6':
        return FusionHead('shift_cos_conv', overlap='fixed_priority')
    if arm in ('R4', 'R5', 'R6'):
        return FusionHead('shift_cos_conv', region_input={
            'R4': 'zero', 'R5': 'normal', 'R6': 'detach'}[arm],
            image_loss=arm != 'R5')
    if arm in ('W4', 'W6', 'W8'):
        return FusionHead('shift_cos_conv', window=int(arm[1:]))
    if arm in ('E0', 'S5', 'F4'):
        return FusionHead('shift_cos_conv')
    if arm in FUSION_ARMS:
        return FusionHead(FUSION_ARMS[arm])
    raise ValueError(f'Unknown BladeGL arm: {arm}')


def required_views(arm: str) -> tuple[bool, bool, bool]:
    """Return include_direct, include_tiles, include_global."""
    if arm not in SUPPORTED_ARMS:
        raise ValueError(f'Unknown BladeGL arm: {arm}')
    if arm == 'S1':
        return False, False, True
    if arm == 'S2':
        return True, False, True
    if arm == 'S3':
        return False, True, True
    return False, True, True


def loss_value(arm, z, score, target, valid, labels):
    bce = F.binary_cross_entropy_with_logits(
        z.float(), target.float(), pos_weight=z.new_tensor(8.).float(), reduction='none')
    bce = (bce * valid).sum() / valid.sum().clamp_min(1)
    prob = z.float().sigmoid() * valid
    abnormal = labels > 0
    dice = 1 - (2 * (prob * target).sum((1, 2)) + 1) / (
        prob.sum((1, 2)) + target.sum((1, 2)) + 1)
    dl = dice[abnormal].mean() if abnormal.any() else z.sum() * 0
    image = F.binary_cross_entropy_with_logits(score.float(), labels)
    return bce + .3 * dl + (0 if arm == 'R5' else .2 * image)
