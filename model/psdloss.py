import torch
import torch.fft
import torch.nn as nn


class PSDLoss(nn.Module):
    """
    Frequency-domain power spectral density loss with dynamic masking.

    The loss compares radially averaged PSD curves. It down-weights frequencies
    that are already exposed by the coarse condition and emphasizes frequencies
    that the model still needs to synthesize.
    """

    def __init__(
        self,
        nbins: int = 64,
        log_spectrum: bool = True,
        normalize: str = "shape",      # "shape" compares PSD distributions; any other value compares raw PSDs.
        high_freq_boost: float = 1.0,   # Extra weighting exponent for high-frequency bins.
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

        # Cache resolution-dependent tensors to avoid rebuilding them every call.
        self.register_buffer("_idx_bins", None, persistent=False)
        self.register_buffer("_bin_counts", None, persistent=False)
        self.register_buffer("_base_weights", None, persistent=False)
        self.register_buffer("_hann2d", None, persistent=False)
        self._cached_hw = None

    @torch.no_grad()
    def _build_cache(self, H: int, W: int, device, dtype):
        if self._cached_hw == (H, W):
            return

        # Build the frequency grid used by the real FFT output.
        fy = torch.fft.fftfreq(H, d=1.0, device=device)
        fx = torch.fft.rfftfreq(W, d=1.0, device=device)
        grid_y, grid_x = torch.meshgrid(fy, fx, indexing="ij")

        # Normalize frequency radii to [0, 1].
        radius = torch.sqrt(grid_x**2 + grid_y**2)
        r_norm = radius / (radius.max() + 1e-12)
        r_flat = r_norm.reshape(-1)

        # Assign each FFT coefficient to a radial bin.
        edges = torch.linspace(0.0, 1.0, self.nbins + 1, device=device)
        idx_bins = torch.bucketize(r_flat, edges, right=False) - 1
        idx_bins = idx_bins.clamp(min=0, max=self.nbins - 1)

        # Count coefficients per bin for the radial average.
        bin_counts = torch.bincount(idx_bins, minlength=self.nbins).to(dtype=dtype)

        # Base weights grow with radius to emphasize high-frequency detail.
        bin_centers = 0.5 * (edges[:-1] + edges[1:])
        base_w = (bin_centers + 0.1).pow(self.high_freq_boost)

        # Windowing reduces spectral leakage from non-periodic image boundaries.
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
        """Optionally compress non-negative precipitation intensity before PSD."""
        if self.apply_log1p:
            x = torch.clamp_min(x, 0.0)
            x = torch.log1p(self.log1p_scale * x)
        return x

    def _psd_radial(self, x2d: torch.Tensor) -> torch.Tensor:
        """Return radially averaged PSD curves with shape [N, nbins]."""
        N, H, W = x2d.shape
        self._build_cache(H, W, x2d.device, x2d.dtype)

        x_windowed = x2d * self._hann2d
        Fk = torch.fft.rfftn(x_windowed, dim=(-2, -1), norm="ortho")
        Pk = Fk.abs().pow(2)

        # Compensate for the missing conjugate half of the spectrum in rFFT.
        if W % 2 == 0:
            Pk[..., 1:-1] *= 2.0
        else:
            Pk[..., 1:] *= 2.0

        P_flat = Pk.reshape(N, -1)
        spec_sum = torch.zeros(N, self.nbins, device=x2d.device, dtype=x2d.dtype)
        spec_sum.scatter_add_(1, self._idx_bins.unsqueeze(0).expand(N, -1), P_flat)

        return spec_sum / (self._bin_counts.unsqueeze(0) + self.eps)

    def forward(self, pred: torch.Tensor, target: torch.Tensor, s_batch: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred, target: Tensors with shape [B, H, W], [B, T, H, W], or [Total_N, H, W].
            s_batch: Tensor with shape [B], usually containing scales in [0, W].
        """
        assert pred.shape == target.shape

        # Flatten time and batch dimensions into a single image dimension.
        if pred.ndim == 4:
            B, T, H, W = pred.shape
            x2d = pred.reshape(B * T, H, W)
            y2d = target.reshape(B * T, H, W)
        elif pred.ndim == 3:
            x2d, y2d = pred, target
            H, W = x2d.shape[-2], x2d.shape[-1]
        else:
            raise ValueError(f"Unsupported input shape: {pred.shape}")

        # Align one coarse-scale value per original sample with all flattened frames.
        total_n = x2d.shape[0]
        actual_b = s_batch.shape[0]

        if total_n % actual_b == 0:
            repeat_factor = total_n // actual_b
            s_val = s_batch.repeat_interleave(repeat_factor)
        else:
            raise ValueError(f"Total samples {total_n} is not divisible by s_batch size {actual_b}")

        spec_x = self._psd_radial(self._preprocess(x2d)) # [Total_N, nbins]
        spec_y = self._psd_radial(self._preprocess(y2d))

        if self.normalize == "shape":
            spec_x = spec_x / (spec_x.sum(dim=1, keepdim=True) + self.eps)
            spec_y = spec_y / (spec_y.sum(dim=1, keepdim=True) + self.eps)

        if self.log_spectrum:
            spec_x = torch.log(spec_x + self.eps)
            spec_y = torch.log(spec_y + self.eps)

        # Map the retained scale to a radial-bin cutoff.
        current_bin_idx = (s_val.float() / W * self.nbins).long().clamp(max=self.nbins - 1)

        bin_indices = torch.arange(self.nbins, device=x2d.device).unsqueeze(0) # [1, nbins]
        target_idx = current_bin_idx.unsqueeze(1) # [Total_N, 1]

        # Frequencies already covered by the coarse condition get lower weight;
        # still-uncovered higher frequencies remain the main generation target.
        mask = torch.where(bin_indices <= target_idx, 0.2, 1.0)

        final_weights = self._base_weights.unsqueeze(0) * mask # [Total_N, nbins]

        diff = (spec_x - spec_y).pow(2)
        sample_loss = (diff * final_weights).sum(dim=1) / (final_weights.sum(dim=1) + self.eps)

        return sample_loss


if __name__ == "__main__":
    torch.manual_seed(0)
    B, T, H, W = 2, 4, 128, 128
    pred = torch.rand(B, T, H, W)
    target = pred.clone() + 0.05 * torch.randn(B, T, H, W)
    s_batch = torch.full((B,), W // 2)

    criterion = PSDLoss(
        nbins=64,               # Suitable for 128x128 to 512x512 images.
        log_spectrum=True,      # Prevents low-frequency energy from dominating high-frequency detail.
        normalize="shape",      # Compares texture distribution; MAE/MSE should handle absolute values.
        high_freq_boost=1.0,    # Linearly emphasizes high frequencies for sharper edges.
        apply_hann_window=True, # Reduces cross-like artifacts from non-periodic boundaries.
        apply_log1p=True,       # Useful for heavy-tailed, non-negative precipitation data.
        log1p_scale=50.0        # Tune this according to the data normalization range.
    )
    loss = criterion(pred, target, s_batch)
    print("PSD loss:", float(loss.mean()))
