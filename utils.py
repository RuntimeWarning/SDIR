import torch
import random
import numpy as np
import torch.nn.functional as F
from collections.abc import Iterable
from torch.utils.checkpoint import checkpoint, checkpoint_sequential



def compute_radial_psd(field):
    """field: (H, W) numpy array"""
    H, W = field.shape
    # Hann 窗减少边缘效应
    window = np.hanning(H)[:, None] * np.hanning(W)[None, :]
    fft = np.fft.rfft2(field * window)
    psd_2d = (np.abs(fft) ** 2) / (H * W)

    # 径向平均
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
    pred, gt: (B, T, H, W) numpy array
    返回每个样本的平均斜率误差
    """
    errors = []
    B, T, H, W = pred.shape
    for b in range(B):
        for t in range(T):
            k_pred, psd_pred = compute_radial_psd(pred[b, t])
            k_gt,   psd_gt   = compute_radial_psd(gt[b, t])

            # 只取湍流惯性子区间（去掉最低和最高频率）
            valid = (k_pred > 0.05) & (k_pred < 0.4)
            log_k = np.log(k_pred[valid])

            # log-log 线性拟合，斜率即为谱斜率
            slope_pred = np.polyfit(log_k, np.log(psd_pred[valid] + 1e-8), 1)[0]
            slope_gt   = np.polyfit(log_k, np.log(psd_gt[valid]   + 1e-8), 1)[0]

            errors.append(abs(slope_pred - slope_gt))

    return np.mean(errors)


def get_coarse_condition(sequence_gt, alpha=1.0, beta=3.0):
    """
    基于频域截断的粗尺度条件生成
    alpha, beta: 控制 Beta 分布的形状。
    默认 alpha=0.8, beta=1.5 使采样峰值集中在低频区域 (s 较小)。
    """
    B, T, C, H, W = sequence_gt.shape
    device = sequence_gt.device
    
    # 1. 使用 Beta 分布生成采样比例 (0~1)
    # alpha < beta 时，分布偏向左侧（低频）
    m = torch.distributions.Beta(torch.tensor([alpha]), torch.tensor([beta]))
    s_ratio = m.sample((B,)).to(device).view(B) # [B]
    
    # 2. 映射到实际的频率尺度 s ∈ [0, W]
    # 这里 s 代表保留左上角 s*s 的 DCT 系数
    s_tensor = torch.round(s_ratio * W).long()
    
    low_res_condition = torch.zeros_like(sequence_gt)

    for i in range(B):
        s = s_tensor[i].item()
        
        if s == 0:
            # s=0 时，低频条件为全黑（或全局均值），模拟完全冷启动
            continue
        elif s >= W:
            # s=W 时，保留全部频率，即原始图像
            low_res_condition[i] = sequence_gt[i]
        else:
            imgs_resize = F.interpolate(sequence_gt[i], size=(s, s), mode='bicubic')
            low_res_condition[i] = F.interpolate(imgs_resize, size=(H, W), mode='bicubic')

    # 将 s 作为标量信号返回，用于 AdaIN 注入
    # 转换为 float 类型以适配后续的线性层处理
    resolution_tensor = s_tensor.to(sequence_gt.dtype) 
    
    return low_res_condition, resolution_tensor


def auto_grad_checkpoint(module, *args, **kwargs):
    if getattr(module, "grad_checkpointing", False):
        if not isinstance(module, Iterable):
            return checkpoint(module, *args, use_reentrant=False, **kwargs)
        gc_step = module[0].grad_checkpointing_step
        return checkpoint_sequential(module, gc_step, *args, use_reentrant=False, **kwargs)
    return module(*args, **kwargs)


def mask_by_order(mask_len, order, bsz, seq_len):
    masking = torch.zeros(bsz, seq_len).cuda()
    masking = torch.scatter(masking, dim=-1, index=order[:, :mask_len.long()], src=torch.ones(bsz, seq_len).cuda())#.bool()
    return masking