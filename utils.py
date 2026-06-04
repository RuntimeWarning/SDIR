import argparse
import torch
import random
import numpy as np
import torch.nn.functional as F
from collections.abc import Iterable
from torch.utils.checkpoint import checkpoint, checkpoint_sequential



def compute_radial_psd(field):
    """Compute the radially averaged power spectral density of a 2D field."""
    H, W = field.shape
    # Use a Hann window to reduce boundary artifacts before the FFT.
    window = np.hanning(H)[:, None] * np.hanning(W)[None, :]
    fft = np.fft.rfft2(field * window)
    psd_2d = (np.abs(fft) ** 2) / (H * W)

    # Average spectral power over radial frequency bins.
    ky = np.fft.fftfreq(H)
    kx = np.fft.rfftfreq(W)
    KX, KY = np.meshgrid(kx, ky)
    K = np.sqrt(KX**2 + KY**2)

    k_bins = np.linspace(0, 0.5, min(H, W) // 2)
    psd_radial = []
    for i in range(len(k_bins) - 1):
        mask = (K >= k_bins[i]) & (K < k_bins[i+1])
        if mask.sum() > 0:
            psd_radial.append(psd_2d[mask].mean())

    return np.array(k_bins[1:len(psd_radial)+1]), np.array(psd_radial)


def spectral_slope_error(pred, gt):
    """
    Compute the mean absolute error between predicted and target PSD slopes.

    Args:
        pred, gt: Arrays with shape (B, T, H, W).
    """
    errors = []
    B, T, H, W = pred.shape
    for b in range(B):
        for t in range(T):
            k_pred, psd_pred = compute_radial_psd(pred[b, t])
            k_gt,   psd_gt   = compute_radial_psd(gt[b, t])

            # Fit only the inertial-like mid-frequency range and skip extremes.
            valid = (k_pred > 0.05) & (k_pred < 0.4)
            log_k = np.log(k_pred[valid])

            # The slope of the log-log fit is the spectral slope.
            slope_pred = np.polyfit(log_k, np.log(psd_pred[valid] + 1e-8), 1)[0]
            slope_gt   = np.polyfit(log_k, np.log(psd_gt[valid]   + 1e-8), 1)[0]

            errors.append(abs(slope_pred - slope_gt))

    return np.mean(errors)


def get_coarse_condition(sequence_gt, alpha=1.0, beta=3.0):
    """
    Generate coarse-scale conditioning by sampling a retained frequency scale.

    Args:
        sequence_gt: Ground-truth sequence with shape (B, T, C, H, W).
        alpha, beta: Shape parameters for the Beta distribution. With
            alpha < beta, sampled scales are biased toward low frequencies.
    """
    B, T, C, H, W = sequence_gt.shape
    device = sequence_gt.device
    
    # Sample a frequency retention ratio in [0, 1].
    m = torch.distributions.Beta(torch.tensor([alpha]), torch.tensor([beta]))
    s_ratio = m.sample((B,)).to(device).view(B) # [B]
    
    # Map the ratio to the spatial scale used by bicubic down/up sampling.
    s_tensor = torch.round(s_ratio * W).long()
    
    low_res_condition = torch.zeros_like(sequence_gt)

    for i in range(B):
        s = s_tensor[i].item()
        
        if s == 0:
            # An all-zero condition simulates a cold start with no coarse signal.
            continue
        elif s >= W:
            # Full scale keeps the original sequence unchanged.
            low_res_condition[i] = sequence_gt[i]
        else:
            imgs_resize = F.interpolate(sequence_gt[i], size=(s, s), mode='bicubic')
            low_res_condition[i] = F.interpolate(imgs_resize, size=(H, W), mode='bicubic')

    # Return the scale as a floating-point conditioning signal for AdaIN layers.
    resolution_tensor = s_tensor.to(sequence_gt.dtype) 
    
    return low_res_condition, resolution_tensor


def auto_grad_checkpoint(module, *args, **kwargs):
    """Run a module with gradient checkpointing when the module enables it."""
    if getattr(module, "grad_checkpointing", False):
        if not isinstance(module, Iterable):
            return checkpoint(module, *args, use_reentrant=False, **kwargs)
        gc_step = module[0].grad_checkpointing_step
        return checkpoint_sequential(module, gc_step, *args, use_reentrant=False, **kwargs)
    return module(*args, **kwargs)


def mask_by_order(mask_len, order, bsz, seq_len):
    """Build a binary mask by selecting the first ``mask_len`` indices in ``order``."""
    masking = torch.zeros(bsz, seq_len).cuda()
    masking = torch.scatter(masking, dim=-1, index=order[:, :mask_len.long()], src=torch.ones(bsz, seq_len).cuda())#.bool()
    return masking

def str2bool(v):
    """Parse common command-line string values into booleans."""
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")
