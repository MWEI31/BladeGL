"""Synchronized, batch-one raw-image inference timing on the active GPU."""
from __future__ import annotations

import time

import numpy as np
import torch


def _exclusive_profile(function):
    from functools import wraps

    @wraps(function)
    def wrapped(*args, **kwargs):
        import os
        descriptor = os.environ.get('BLADEGL_EVAL_LOCK_FD')
        if descriptor is None:
            return function(*args, **kwargs)
        import fcntl
        fd = int(descriptor)
        print('COST_WAIT_EXCLUSIVE', flush=True)
        fcntl.flock(fd, fcntl.LOCK_UN)
        fcntl.flock(fd, fcntl.LOCK_EX)
        print('COST_PROFILE_EXCLUSIVE', flush=True)
        try:
            return function(*args, **kwargs)
        finally:
            print('COST_PROFILE_DONE', flush=True)
            fcntl.flock(fd, fcntl.LOCK_SH)
    return wrapped


@_exclusive_profile
def profile_rgb_path(rows, run_one, modules, device, warmup=5, measured=32):
    """run_one(row) must read the RGB file and return a map plus image score."""
    if not rows:
        raise ValueError('No rows available for timing')
    if device.type != 'cuda' or not torch.cuda.is_available():
        raise RuntimeError('Cost profiling requires the active CUDA GPU')
    sample = [rows[int(i)] for i in np.linspace(0, len(rows)-1,
                                               min(measured, len(rows)), dtype=int)]
    for i in range(warmup):
        run_one(sample[i % len(sample)])
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    times = []
    for row in sample:
        torch.cuda.synchronize(device)
        before = time.perf_counter_ns()
        run_one(row)
        torch.cuda.synchronize(device)
        times.append((time.perf_counter_ns()-before)/1e6)
    unique = {id(m): m for m in modules if isinstance(m, torch.nn.Module)}
    parameters = [p for module in unique.values() for p in module.parameters()]
    return {
        'scope': 'RGB file decode through preprocessing, H2D, encoding, method, '
                 'grid projection and score; excludes metric calculation and map save',
        'batch_size': 1, 'warmup': warmup, 'measured_images': len(sample),
        'sample_ids': [r['sample_id'] for r in sample],
        'precision': 'runner_native',
        'p50_ms': float(np.percentile(times, 50)),
        'p95_ms': float(np.percentile(times, 95)),
        'mean_ms': float(np.mean(times)),
        'per_image_ms': times,
        'peak_vram_mib': float(torch.cuda.max_memory_allocated(device)/2**20),
        'total_parameters': sum(p.numel() for p in parameters),
        'trainable_parameters': sum(p.numel() for p in parameters if p.requires_grad),
    }
