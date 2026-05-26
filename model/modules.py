import math
import torch
from torch import Tensor, nn
from dataclasses import dataclass
from einops import rearrange, repeat
from liger_kernel.ops.rope import LigerRopeFunction
from liger_kernel.ops.rms_norm import LigerRMSNormFunction
from flash_attn import flash_attn_func as flash_attn_func_v2
try:
    from flash_attn_interface import flash_attn_func as flash_attn_func_v3
    SUPPORT_FA3 = True
except:
    SUPPORT_FA3 = False



def prepare_ids(bs, t, h, w, device, dtype):
    img_ids = torch.zeros(t, h, w, 3)
    img_ids[..., 0] = img_ids[..., 0] + torch.arange(t)[:, None, None]
    img_ids[..., 1] = img_ids[..., 1] + torch.arange(h)[None, :, None]
    img_ids[..., 2] = img_ids[..., 2] + torch.arange(w)[None, None, :]
    img_ids = repeat(img_ids, "t h w c -> b (t h w) c", b=bs)

    return img_ids.to(device, dtype)


def liger_rope(pos: Tensor, dim: int, theta: int):
    assert dim % 2 == 0
    scale = torch.arange(0, dim, 2, dtype=torch.float32, device=pos.device) / dim
    omega = 1.0 / (theta**scale)
    out = torch.einsum("...n,d->...nd", pos, omega)  # (b, seq, dim//2)
    cos = out.cos()
    sin = out.sin()
    return (cos, sin)


class LigerEmbedND(nn.Module):
    def __init__(self, theta: int, axes_dim: list[int]):
        super().__init__()
        self.theta = theta
        self.axes_dim = axes_dim

    def forward(self, ids: Tensor) -> Tensor:
        n_axes = ids.shape[-1]
        cos_list = []
        sin_list = []
        for i in range(n_axes):
            cos, sin = liger_rope(ids[..., i], self.axes_dim[i], self.theta)
            cos_list.append(cos)
            sin_list.append(sin)
        cos_emb = torch.cat(cos_list, dim=-1).repeat(1, 1, 2).contiguous()
        sin_emb = torch.cat(sin_list, dim=-1).repeat(1, 1, 2).contiguous()

        return (cos_emb, sin_emb)


class LastLayer(nn.Module):
    def __init__(self, hidden_size: int, patch_size: int, out_channels: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True))

    def forward(self, x: Tensor, vec: Tensor) -> Tensor:
        shift, scale = self.adaLN_modulation(vec).chunk(2, dim=1)
        x = (1 + scale[:, None, :]) * self.norm_final(x) + shift[:, None, :]
        x = self.linear(x)
        return x


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor):
        x_dtype = x.dtype
        x = x.float()
        rrms = torch.rsqrt(torch.mean(x**2, dim=-1, keepdim=True) + 1e-6)
        return (x * rrms).to(dtype=x_dtype) * self.scale


class FusedRMSNorm(RMSNorm):
    def forward(self, x: Tensor):
        return LigerRMSNormFunction.apply(x, self.scale, 1e-6, 0.0, "llama", False)


@dataclass
class ModulationOut:
    shift: Tensor
    scale: Tensor
    gate: Tensor


class Modulation(nn.Module):
    def __init__(self, dim: int, double: bool):
        super().__init__()
        self.is_double = double
        self.multiplier = 6 if double else 3
        self.lin = nn.Linear(dim, self.multiplier * dim, bias=True)

    def forward(self, vec: Tensor) -> tuple[ModulationOut, ModulationOut | None]:
        out = self.lin(nn.functional.silu(vec))[:, None, :].chunk(self.multiplier, dim=-1)

        return (
            ModulationOut(*out[:3]),
            ModulationOut(*out[3:]) if self.is_double else None,
        )


class QKNorm(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.query_norm = FusedRMSNorm(dim)
        self.key_norm = FusedRMSNorm(dim)

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        q = self.query_norm(q)
        k = self.key_norm(k)
        return q.to(v), k.to(v)


def flash_attn_func(q: Tensor, k: Tensor, v: Tensor) -> Tensor:
    if SUPPORT_FA3:
        return flash_attn_func_v3(q, k, v) #[0]
    return flash_attn_func_v2(q, k, v)


def attention(q: Tensor, k: Tensor, v: Tensor, pe, dtype=torch.float16) -> Tensor:

    cos, sin = pe
    q, k = LigerRopeFunction.apply(q, k, cos, sin)

    # params
    out_dtype = q.dtype

    half_dtypes = (torch.float16, torch.bfloat16)

    def half(x):
        return x if x.dtype in half_dtypes else x.to(dtype)

    # preprocess query, key, value
    q = half(q)
    k = half(k)
    v = half(v)

    q = rearrange(q, "B H L D -> B L H D")
    k = rearrange(k, "B H L D -> B L H D")
    v = rearrange(v, "B H L D -> B L H D")
    x = flash_attn_func(q, k, v)
    x = rearrange(x, "B L H D -> B L (H D)")
    return x.type(out_dtype)


class SingleStreamBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        fused_qkv: bool = True
    ):
        super().__init__()
        self.hidden_dim = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.fused_qkv = fused_qkv
        self.mlp_hidden_dim = int(hidden_size * mlp_ratio)
        if fused_qkv:
            # qkv and mlp_in
            self.linear1 = nn.Linear(hidden_size, hidden_size * 3 + self.mlp_hidden_dim)
        else:
            self.q_proj = nn.Linear(hidden_size, hidden_size)
            self.k_proj = nn.Linear(hidden_size, hidden_size)
            self.v_mlp = nn.Linear(hidden_size, hidden_size + self.mlp_hidden_dim)
        # proj and mlp_out
        self.linear2 = nn.Linear(hidden_size + self.mlp_hidden_dim, hidden_size)
        self.norm = QKNorm(self.head_dim)
        self.hidden_size = hidden_size
        self.pre_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp_act = nn.GELU(approximate="tanh")
        self.modulation = Modulation(hidden_size, double=False)

    def forward(self, x: Tensor, vec: Tensor, pe: Tensor) -> Tensor:
        mod, _ = self.modulation(vec)
        x_mod = (1 + mod.scale) * self.pre_norm(x) + mod.shift
        if self.fused_qkv:
            qkv, mlp = torch.split(self.linear1(x_mod), [3 * self.hidden_size, self.mlp_hidden_dim], dim=-1)
            q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        else:
            q = rearrange(self.q_proj(x_mod), "B L (H D) -> B L H D", H=self.num_heads)
            k = rearrange(self.k_proj(x_mod), "B L (H D) -> B L H D", H=self.num_heads)
            v, mlp = torch.split(self.v_mlp(x_mod), [self.hidden_size, self.mlp_hidden_dim], dim=-1)
            v = rearrange(v, "B L (H D) -> B L H D", H=self.num_heads)
        q, k = self.norm(q, k, v)
        if not self.fused_qkv:
            q = rearrange(q, "B L H D -> B H L D")
            k = rearrange(k, "B L H D -> B H L D")
            v = rearrange(v, "B L H D -> B H L D")
        # compute attention
        attn = attention(q, k, v, pe=pe)
        # compute activation in mlp stream, cat again and run second linear layer
        output = self.linear2(torch.cat((attn, self.mlp_act(mlp)), 2))
        output = x + mod.gate * output
        return output


class FrequencyScaleEmbedder(nn.Module):

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def scale_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal scale embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half)
        freqs = freqs.to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.scale_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb