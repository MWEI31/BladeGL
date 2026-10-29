"""Checkpoint identity and calibration validation."""
import math

from .protocol import resolve, sha256


def validate_run_lock(recorded, current):
    changed = sorted(key for key in set(recorded) | set(current)
                     if recorded.get(key) != current.get(key))
    if changed:
        raise ValueError('Training inputs changed: ' + ', '.join(changed))


def model_identity(path, checkpoint, cfg, arm):
    lock = checkpoint.get('lock')
    required = ('dataset', 'arm', 'seed', 'budget', 'config_sha256',
                'manifest_sha256', 'backbone_sha256')
    if not isinstance(lock, dict) or any(key not in lock for key in required):
        raise ValueError('Checkpoint is missing training metadata.')
    if lock['arm'] != arm:
        raise ValueError('Checkpoint arm does not match --arm.')
    if lock['config_sha256'] != cfg['_config_sha256']:
        raise ValueError('Configuration does not match the checkpoint.')
    backbone_hash = sha256(resolve(cfg['dino_weights']))
    if lock['backbone_sha256'] != backbone_hash:
        raise ValueError('Backbone weights do not match the checkpoint.')
    return {**{key: lock[key] for key in required},
            'checkpoint_sha256': sha256(path)}


def validate_calibration(calibration, identity, mode, weight):
    if calibration.get('model') != identity:
        raise ValueError('Calibration does not match this model; regenerate it with evaluate.')
    if calibration.get('readout') != mode or calibration.get('head_weight') != weight:
        raise ValueError('Calibration readout does not match the requested readout.')
    for key in ('image_threshold', 'pixel_threshold'):
        value = calibration.get(key)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError('Invalid calibration threshold: ' + key)
