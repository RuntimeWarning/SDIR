import torch
import torch.fft
import torch.nn as nn



class PSDLoss(nn.Module):
    """
    频域引导的功率谱密度损失函数 (Physical-Consistent & Dynamic Masking)
    """
    def __init__(
        self,
        nbins: int = 64,
        log_spectrum: bool = True,
        normalize: str = "shape",      # "shape" (关注分布) | "none" (关注绝对值)
        high_freq_boost: float = 1.0,   # 对高频部分的额外加权指数
        apply_hann_window: bool = True,
        apply_log1p: bool = True,
        log1p_scale: float = 50.0,
        eps: float = 1e-8
    ):
        super().__init__()
        self.nbins = nbins
        self.log_spectrum = log_spectrum
        self.normalize = normalize
        self.high_freq_boost = high_freq_boost
        self.apply_hann_window = apply_hann_window
        self.apply_log1p = apply_log1p
        self.log1p_scale = log1p_scale
        self.eps = eps

        # 缓存机制，避免重复计算
        self.register_buffer("_idx_bins", None, persistent=False)
        self.register_buffer("_bin_counts", None, persistent=False)
        self.register_buffer("_base_weights", None, persistent=False)
        self.register_buffer("_hann2d", None, persistent=False)
        self._cached_hw = None

    @torch.no_grad()
    def _build_cache(self, H: int, W: int, device, dtype):
        if self._cached_hw == (H, W):
            return

        # 1. 构造频率格点 (针对 rfft)
        fy = torch.fft.fftfreq(H, d=1.0, device=device)
        fx = torch.fft.rfftfreq(W, d=1.0, device=device)
        grid_y, grid_x = torch.meshgrid(fy, fx, indexing="ij")
        
        # 归一化频率半径 (0 to 1)
        radius = torch.sqrt(grid_x**2 + grid_y**2)
        r_norm = radius / (radius.max() + 1e-12)
        r_flat = r_norm.reshape(-1)

        # 2. 径向分桶 (Radial Binning)
        edges = torch.linspace(0.0, 1.0, self.nbins + 1, device=device)
        idx_bins = torch.bucketize(r_flat, edges, right=False) - 1
        idx_bins = idx_bins.clamp(min=0, max=self.nbins - 1)

        # 3. 统计每个桶的像素数
        bin_counts = torch.bincount(idx_bins, minlength=self.nbins).to(dtype=dtype)
        
        # 4. 高频增强基础权重
        bin_centers = 0.5 * (edges[:-1] + edges[1:])
        base_w = (bin_centers + 0.1).pow(self.high_freq_boost)

        # 5. Hann 窗
        if self.apply_hann_window:
            wy = torch.hann_window(H, device=device, dtype=dtype)
            wx = torch.hann_window(W, device=device, dtype=dtype)
            hann2d = torch.outer(wy, wx)
        else:
            hann2d = torch.ones(H, W, device=device, dtype=dtype)

        self._idx_bins = idx_bins
        self._bin_counts = bin_counts
        self._base_weights = base_w
        self._hann2d = hann2d
        self._cached_hw = (H, W)

    def _preprocess(self, x: torch.Tensor) -> torch.Tensor:
        if self.apply_log1p:
            x = torch.clamp_min(x, 0.0)
            x = torch.log1p(self.log1p_scale * x)
        return x

    def _psd_radial(self, x2d: torch.Tensor) -> torch.Tensor:
        N, H, W = x2d.shape
        self._build_cache(H, W, x2d.device, x2d.dtype)

        # 加窗并执行实数 FFT
        x_windowed = x2d * self._hann2d
        Fk = torch.fft.rfftn(x_windowed, dim=(-2, -1), norm="ortho")
        Pk = Fk.abs().pow(2)

        # 物理一致性补偿：补偿 rfft 丢弃的一半对称频谱能量
        if W % 2 == 0:
            Pk[..., 1:-1] *= 2.0
        else:
            Pk[..., 1:] *= 2.0

        # 径向平均
        P_flat = Pk.reshape(N, -1)
        spec_sum = torch.zeros(N, self.nbins, device=x2d.device, dtype=x2d.dtype)
        spec_sum.scatter_add_(1, self._idx_bins.unsqueeze(0).expand(N, -1), P_flat)
        
        return spec_sum / (self._bin_counts.unsqueeze(0) + self.eps)

    def forward(self, pred: torch.Tensor, target: torch.Tensor, s_batch: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred, target: [B, H, W] 或 [B, T, H, W] 或 [Total_N, H, W]
            s_batch: [B] 的 Tensor, 范围通常在 [0, W]
        """
        assert pred.shape == target.shape
        
        # 1. 统一形状为 [Total_N, H, W]
        if pred.ndim == 4:
            B, T, H, W = pred.shape
            x2d = pred.reshape(B * T, H, W)
            y2d = target.reshape(B * T, H, W)
        elif pred.ndim == 3:
            x2d, y2d = pred, target
            H, W = x2d.shape[-2], x2d.shape[-1]
        else:
            raise ValueError(f"Unsupported input shape: {pred.shape}")

        # 2. 动态对齐 s_batch 到 Total_N (重点修复报错)
        total_n = x2d.shape[0]
        actual_b = s_batch.shape[0]
        
        if total_n % actual_b == 0:
            repeat_factor = total_n // actual_b
            # 使用 repeat_interleave 确保每个样本对应的 s 信号正确
            s_val = s_batch.repeat_interleave(repeat_factor)
        else:
            raise ValueError(f"Total samples {total_n} is not divisible by s_batch size {actual_b}")

        # 3. 预处理与频谱计算
        spec_x = self._psd_radial(self._preprocess(x2d)) # [Total_N, nbins]
        spec_y = self._psd_radial(self._preprocess(y2d))

        # 4. 归一化与对数域转换
        if self.normalize == "shape":
            spec_x = spec_x / (spec_x.sum(dim=1, keepdim=True) + self.eps)
            spec_y = spec_y / (spec_y.sum(dim=1, keepdim=True) + self.eps)

        if self.log_spectrum:
            spec_x = torch.log(spec_x + self.eps)
            spec_y = torch.log(spec_y + self.eps)

        # 5. 动态频率掩码逻辑 (从粗到精引导)
        # 将 s_val [0, W] 映射到归一化的 bin 索引
        # 这里的映射需要与 _build_cache 里的 r_norm 对齐
        current_bin_idx = (s_val.float() / W * self.nbins).long().clamp(max=self.nbins-1)
        
        bin_indices = torch.arange(self.nbins, device=x2d.device).unsqueeze(0) # [1, nbins]
        target_idx = current_bin_idx.unsqueeze(1) # [Total_N, 1]

        # 掩码策略：
        # 已解锁的低频部分 (<= target_idx) 权重设小 (0.2)，模型已通过条件注入学习到
        # 未解锁的高频部分 (> target_idx) 权重设大 (1.0)，是当前迭代生成的重点
        mask = torch.where(bin_indices <= target_idx, 0.2, 1.0)

        # 6. 计算加权损失
        # 组合基础高频增强与动态掩码
        final_weights = self._base_weights.unsqueeze(0) * mask # [Total_N, nbins]
        
        # 使用 MSE 刻画谱线差异
        diff = (spec_x - spec_y).pow(2)
        
        # 归一化加权平均
        sample_loss = (diff * final_weights).sum(dim=1) / (final_weights.sum(dim=1) + self.eps)

        return sample_loss


if __name__ == "__main__":
    # 简单自测
    torch.manual_seed(0)
    B, T, H, W = 2, 4, 128, 128
    pred = torch.rand(B, T, H, W)
    target = pred.clone() + 0.05 * torch.randn(B, T, H, W)

    criterion = PSDLoss(nbins=64,               # 适配 128x128 到 512x512 的图像
                        log_spectrum=True,      # 必须开启，否则低频能量会淹没高频细节
                        normalize="shape",      # 核心推荐：只学“纹理分布”，把数值准确性交给 MSE/MAE
                        high_freq_boost=1.0,    # 线性增强高频，强制模型生成锐利边缘
                        apply_hann_window=True, # 必须开启，消除非周期性边界造成的十字伪影
                        apply_log1p=True,       # 降雨数据强烈建议开启
                        log1p_scale=50.0         # 需要根据你的数据归一化方式调整（见下方详述）
                    )
    l = criterion(pred, target)
    print("PSD loss:", float(l))

'''
参数说明
- nbins=64
  - 作用：将2D频谱按“半径”（频率大小）做径向分桶的数量。越大，频率分辨率越细，但更嘈杂、计算略高。
  - 何时调大：分辨率较高（≥128×128）、希望更精细地区分尺度。
  - 何时调小：样本尺寸较小、谱曲线抖动明显时。
  - 建议范围：32–96（128×128 时 64 合理；256×256 可用 64–128）。

- log_spectrum=True
  - 作用：在对数域比较谱，抑制低频大能量的主导，提升对高频差异的敏感度；把“乘性偏差”转化为“加性差异”来度量。
  - 何时关闭：你非常关心绝对能量差（而非形状），或数据中确有大量精确的零值、想避免任何对数处理（实现里已用 eps 稳定）。
  - 默认建议：开启。

- normalize="shape"
  - 作用：控制谱的归一化方式。
    - "shape"：把每个样本的谱归一化为和为1，只比较谱形状（能量分布），忽略总能量差。常与 MSE 搭配，MSE 负责幅值，PSD 负责结构。
    - "power"：将预测的总谱功率按目标的总功率缩放后再比较。一定程度保留尺度信息，但弱化幅值差异对 PSD 的影响。
    - "none"：不做归一化，直接比绝对谱。容易被低频/幅值主导，一般不推荐，除非你确实要让 PSD 强烈约束能量幅度。
  - 默认建议："shape"。

- high_freq_boost=1.0
  - 作用：对高频桶加权，权重随频率半径 r 增长，近似 \(w(r) \propto r^\alpha\)，其中 \(\alpha=\) high_freq_boost。
  - 影响：α 越大，越强调细节与边缘，能有效对抗过度平滑；但 α 过大可能放大噪声/斑点。
  - 建议范围：0–2。起步 1.0；如果仍然偏平滑，可试 1.5–2；若出现斑点噪声，降到 0.5–1。

- apply_hann_window=True
  - 作用：在做 FFT 前乘以 2D Hann 窗，降低边界截断造成的谱泄漏，使径向平均更稳定。
  - 何时关闭：你的数据本身周期性良好，或你使用了反射/周期填充能有效消除边缘效应。
  - 默认建议：开启。

- apply_log1p=True
  - 作用：对输入做 log1p(scale·x) 压缩动态范围，特别适合降雨这类重尾、非负变量，避免大雨值完全主导频域对比，让结构差异更可见。
  - 注意：仅适用于非负数据。若变量可为负或已标准化到零均值，建议关闭或改成对称变换。
  - 默认建议：对降雨开启。

- log1p_scale=50.0
  - 作用：log1p 的缩放系数 k，即使用 log1p(k·x)。k 越大：
    - 对小值更“放大”（提升相对权重），对大值更“压缩”（减小主导性）；
    - 可以让小雨/中雨的结构在 PSD 中更有存在感，但过大可能让毛毛雨噪声被过度强调。
  - 选取思路：
    - 如果你的 x 已缩放到 [0,1]，k 通常取 10–50。
    - 若 x 是毫米/小时且范围 0–20，常先把 x/20 归一化到 [0,1]，再用 k=10–20；或直接用 k≈1–5。
  - 调太大：容易放大小雨噪声；调太小：强降雨依然主导，PSD 对结构不敏感。

快速调参建议
- 结果仍过度平滑：适当增大 high_freq_boost 到 1.5–2；或提高 PSD loss 的权重系数 λ；nbins 可略增大（如 64→96）。
- 出现斑点噪声/颗粒感：降低 high_freq_boost（如 1.0→0.5），或小幅减小 λ；如使用 normalize="none"，可改回 "shape" 或 "power"。
- 小雨结构太弱、强降雨主导：增大 log1p_scale（如 20→50）；同时保留 log_spectrum=True。
- 维度较小或谱曲线不稳：减小 nbins（如 64→32），或保持 apply_hann_window=True 以降低谱泄漏。

'''