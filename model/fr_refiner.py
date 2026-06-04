import torch
import torch.nn as nn
import torch.nn.functional as F
from model.modules import FrequencyScaleEmbedder



def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(-1).unsqueeze(-1)) + shift.unsqueeze(-1).unsqueeze(-1)


class SFNO(nn.Module):
    """
    hidden_size: channel dimension size
    num_blocks: how many blocks to use in the block diagonal weight matrices (higher => less complexity but less parameters)
    sparsity_threshold: lambda for softshrink
    hard_thresholding_fraction: how many frequencies you want to completely mask out (lower => hard_thresholding_fraction^2 less FLOPs)
    """
    def __init__(self, hidden_size, num_blocks=8, sparsity_threshold=0.01, hard_thresholding_fraction=1, hidden_size_factor=4):
        super().__init__()
        assert hidden_size % num_blocks == 0, f"hidden_size {hidden_size} should be divisble by num_blocks {num_blocks}"

        self.hidden_size = hidden_size
        self.sparsity_threshold = sparsity_threshold
        self.num_blocks = num_blocks
        self.block_size = self.hidden_size // self.num_blocks
        self.hard_thresholding_fraction = hard_thresholding_fraction
        self.hidden_size_factor = hidden_size_factor
        self.scale = 0.02

        self.w1 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size, self.block_size * self.hidden_size_factor))
        self.b1 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size * self.hidden_size_factor))
        self.w2 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size * self.hidden_size_factor, self.block_size))
        self.b2 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size))

        self.norm = LayerNorm2d(hidden_size, affine=False, eps=1e-6) # Non-affine normalization; modulation supplies shift and scale.
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True) # Predict shift and scale.
        )

    def forward(self, x, s):
        bias = x

        shift, scale = self.adaLN_modulation(s).chunk(2, dim=1)
        x = modulate(self.norm(x), shift, scale)

        dtype = x.dtype
        x = x.float()
        B, C, H, W = x.shape

        x = x.permute(0, 2, 3, 1)
        x = torch.fft.rfft2(x, dim=(1, 2), norm="ortho")
        x = x.reshape(B, x.shape[1], x.shape[2], self.num_blocks, self.block_size)

        o1_real = torch.zeros([B, x.shape[1], x.shape[2], self.num_blocks, self.block_size * self.hidden_size_factor], device=x.device)
        o1_imag = torch.zeros([B, x.shape[1], x.shape[2], self.num_blocks, self.block_size * self.hidden_size_factor], device=x.device)
        o2_real = torch.zeros(x.shape, device=x.device)
        o2_imag = torch.zeros(x.shape, device=x.device)

        total_modes = W // 2 + 1
        kept_modes = int(total_modes * self.hard_thresholding_fraction)

        o1_real[:, :, :kept_modes] = F.relu(
            torch.einsum('...bi,bio->...bo', x[:, :, :kept_modes].real, self.w1[0]) - \
            torch.einsum('...bi,bio->...bo', x[:, :, :kept_modes].imag, self.w1[1]) + \
            self.b1[0]
        )

        o1_imag[:, :, :kept_modes] = F.relu(
            torch.einsum('...bi,bio->...bo', x[:, :, :kept_modes].imag, self.w1[0]) + \
            torch.einsum('...bi,bio->...bo', x[:, :, :kept_modes].real, self.w1[1]) + \
            self.b1[1]
        )

        o2_real[:, :, :kept_modes] = (
            torch.einsum('...bi,bio->...bo', o1_real[:, :, :kept_modes], self.w2[0]) - \
            torch.einsum('...bi,bio->...bo', o1_imag[:, :, :kept_modes], self.w2[1]) + \
            self.b2[0]
        )

        o2_imag[:, :, :kept_modes] = (
            torch.einsum('...bi,bio->...bo', o1_imag[:, :, :kept_modes], self.w2[0]) + \
            torch.einsum('...bi,bio->...bo', o1_real[:, :, :kept_modes], self.w2[1]) + \
            self.b2[1]
        )

        x = torch.stack([o2_real, o2_imag], dim=-1)
        x = F.softshrink(x, lambd=self.sparsity_threshold)
        x = torch.view_as_complex(x)
        x = x.reshape(B, x.shape[1], x.shape[2], C)
        x = torch.fft.irfft2(x, s=(H, W), dim=(1, 2), norm="ortho")
        x = x.permute(0, 3, 1, 2)
        x = x.type(dtype)
        return x + bias


class LayerNorm2d(nn.LayerNorm):
    def __init__(self, num_channels, eps=1e-6, affine=True):
        super().__init__(num_channels, eps=eps, elementwise_affine=affine)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2)
        return x


class GroupNorm(nn.Module):
    def __init__(self, num_channels, num_groups=32, min_channels_per_group=4, eps=1e-5):
        super().__init__()
        self.num_groups = min(num_groups, num_channels // min_channels_per_group)
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))

    def forward(self, x):
        return F.group_norm(x, num_groups=self.num_groups, weight=self.weight.to(x.dtype), bias=self.bias.to(x.dtype), eps=self.eps)
    
    def extra_repr(self) -> str:
        return f"dim={self.weight.numel()}; group={self.num_groups}"


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.norm_final = LayerNorm2d(hidden_size, affine=False, eps=1e-6)
        self.out_proj = nn.Conv2d(hidden_size, out_channels, kernel_size=3, stride=1, padding=1, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.out_proj(x)
        return x

class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x


class Downsample(nn.Module):
    def __init__(self, n_feat, out_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, out_feat // 4, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat, out_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, out_feat * 4, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)
    

class UNetBlock(torch.nn.Module):
    def __init__(self,
        in_channels, out_channels, emb_channels=None, dropout=0, 
        eps=1e-5, actfunc='silu', actf=[1,1], affinef=2, 
        actinada=0, init_zero=0, **kwargs
    ):
        super().__init__()
        self.dropout = dropout
        self.affinef = affinef
        self.norm0 = GroupNorm(num_channels=in_channels, eps=eps, num_groups=kwargs.get('num_groups', 32), min_channels_per_group=kwargs.get('min_channels', 4))
        self.conv0 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.affine = nn.Sequential(
            nn.SiLU() if actinada else nn.Identity(),
            nn.Linear(in_features=emb_channels, out_features=out_channels*self.affinef, bias=True)
        )

        if init_zero:
            nn.init.constant_(self.affine[-1].weight, 0)
            nn.init.constant_(self.affine[-1].bias, 0)

        self.norm1 = GroupNorm(num_channels=out_channels, eps=eps)
        self.conv1 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.skip = None
        self.act = nn.GELU if actfunc=='gelu' else nn.SiLU
        self.act0 = self.act() if actf[0] else nn.Identity()
        self.act1 = self.act() if actf[1] else nn.Identity()
        if out_channels != in_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x, emb):
        orig = x
        x = self.conv0(self.act0(self.norm0(x)))
        params = self.affine(emb).unsqueeze(2).unsqueeze(3).to(x.dtype)
        if self.affinef == 2:
            scale, shift = params.chunk(chunks=2, dim=1)
            gate = 1
        elif self.affinef == 3:
            gate, scale, shift = params.chunk(chunks=3, dim=1)
        x = self.act1(torch.addcmul(shift, self.norm1(x), scale + 1))
        x = self.conv1(F.dropout(x, p=self.dropout, training=self.training))
        x = (gate*x).add_(self.skip(orig) if self.skip is not None else orig)
        return x


class FR_Refiner(nn.Module):
    def __init__(
        self,
        in_channels=5,
        out_channels=10,
        hidden_size=512,
        mult_channels=[1,2,4],
        depth=8,
        **kwargs
    ):
        super().__init__()

        self.levels = 2

        self.x_embedder = OverlapPatchEmbed(in_channels, hidden_size*mult_channels[0], bias=True)
        self.s_embedder_ls = nn.ModuleList([FrequencyScaleEmbedder(hidden_size*mult) for mult in mult_channels[:self.levels+1]])
        self.enc_blocks = nn.ModuleList()
        self.lat_blocks = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()

        # encoder
        for mult, next_mult in zip(mult_channels[:self.levels], mult_channels[1:self.levels+1]):
            channel_size = int(hidden_size * mult)
            self.enc_blocks.append(UNetBlock(channel_size, channel_size, channel_size, **kwargs))
            self.downs.append(Downsample(channel_size, hidden_size * next_mult))

        # latent
        channel_size = int(hidden_size * mult_channels[self.levels])
        for _ in range(depth):
            self.lat_blocks.append(SFNO(channel_size))

        # decoder
        for mult, skip_mult in zip(mult_channels[-self.levels:][::-1], mult_channels[-2::-1]):
            self.ups.append(Upsample(channel_size, hidden_size * mult))
            in_channel_size = hidden_size * mult + hidden_size * skip_mult
            channel_size = int(hidden_size * mult)
            self.dec_blocks.append(UNetBlock(in_channel_size, channel_size, channel_size, **kwargs))

        self.output = nn.Conv2d(channel_size, channel_size, kernel_size=3, stride=1, padding=1, bias=True)
        self.final_layer = FinalLayer(channel_size, out_channels)

    def forward(self, x, s):

        x = self.x_embedder(x)                   # (N, C, H, W)
        c_ls = list() # Multi-resolution scale-conditioning embeddings.

        for idx in range(self.levels+1):
            s_emb = self.s_embedder_ls[idx](s)    # (N, C)
            c_ls.append(s_emb)
        
        skip = list()
        stage_idx = 0

        # Encoder blocks refine features before each downsampling step.
        for idx, block in enumerate(self.enc_blocks):
            x = block(x, c_ls[stage_idx])
            skip.append(x)
            stage_idx += 1
            x = self.downs[idx](x)

        latent_cond = c_ls[self.levels]
        for block in self.lat_blocks:
            x = block(x, latent_cond)

        # Decoder blocks upsample, merge the skip feature, then refine.
        dec_c_ls = c_ls[-1:0:-1]
        for idx, block in enumerate(self.dec_blocks):
            x = self.ups[idx](x)
            x = block(torch.cat([x, skip.pop()], 1), dec_c_ls[idx])

        x = self.output(x)

        x = self.final_layer(x, c_ls[1]) # Use the final decoder-scale conditioning.

        return x



if __name__=="__main__":

    model = FR_Refiner(hidden_size=16, mult_channels=[1,2,4], actf=[1,1], norm_type='gnorm', 
                    norm_type1='gnorm', actinada=1, affinef=3, actfunc='gelu', affine=0,
                    num_groups=16, min_channels=4, in_channels=5,out_channels=10)

    # model.cuda()
    model.eval()

    inputs = torch.rand(2, 5, 128, 128)#.cuda()
    t = torch.ones(2).int()#.cuda()
    out = model(inputs, t)
    print(out.shape)
    params = 0
    for P in model.parameters():
        params += P.numel()

    print(f'PARAMS: {params/1e6:.2f} M')
