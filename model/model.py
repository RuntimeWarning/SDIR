import torch
import torch.nn as nn
from einops import rearrange
from utils import auto_grad_checkpoint
from model.fr_refiner import FR_Refiner
from model.modules import SingleStreamBlock, FrequencyScaleEmbedder, LigerEmbedND, prepare_ids, LastLayer



class Network(nn.Module):

    def __init__(self, configs, latent_size, 
                 hidden_size=1024, num_heads=8, depth=8):
        super().__init__()

        self.configs = configs
        self.latent_size = latent_size
        self.pe_embedder = LigerEmbedND(theta=10000, axes_dim=[44, 42, 42])
        self.index_cond_embed = FrequencyScaleEmbedder(hidden_size)
        token_embed_dim = configs.img_channel * configs.patch_size**2
        self.condition_proj = nn.Linear(token_embed_dim, hidden_size, bias=True)
        self.input_embed = nn.Linear(token_embed_dim, hidden_size, bias=True)
        self.blocks = nn.ModuleList([SingleStreamBlock(hidden_size, num_heads) for _ in range(depth)])
        self.output_layer = LastLayer(hidden_size, self.configs.patch_size, self.configs.img_channel)
        self.fr_refiner = FR_Refiner(hidden_size=32, mult_channels=[1,2,4], actf=[1,1], depth = depth,
                                     actinada=1, affinef=3, actfunc='gelu', num_groups=8, min_channels=4, 
                                     in_channels=configs.input_length+configs.output_length, 
                                     init_zero=1, out_channels=configs.output_length)
        
    def patchify(self, x):
        bsz, t, c, h, w = x.shape
        p = self.configs.patch_size
        h_, w_ = h // p, w // p

        x = x.reshape(bsz, t, c, h_, p, w_, p)
        x = torch.einsum('ntchpwq->nthwcpq', x)
        x = x.reshape(bsz, t * h_ * w_, c * p ** 2)
        return x  # [n, l, d]

    def unpatchify(self, x):
        """ 
            Args:
                x (torch.Tensor): of shape [B, N, C]
            Return:
                x (torch.Tensor): of shape [B, T, C_out, H, W]
        """
        x = rearrange(x,
                      "B (T H W) (C_out H_p W_p) -> B T C_out (H H_p) (W W_p)",
                      T=self.configs.input_length + self.configs.output_length, 
                      H=self.latent_size, W=self.latent_size, 
                      H_p=self.configs.patch_size, 
                      W_p=self.configs.patch_size, 
                      C_out=self.configs.img_channel)
        return x

    def forward(self, inputs, condition, index):
        index_embedding = self.index_cond_embed(index) # N, C
        condition_embedding = self.condition_proj(self.patchify(condition))
        input_embedding = self.input_embed(self.patchify(inputs))
        embedding = torch.cat((input_embedding, condition_embedding), dim=1)
        seq_ids = prepare_ids(self.configs.batch_size, self.configs.input_length + self.configs.output_length, 
                              self.latent_size, self.latent_size, inputs.device, inputs.dtype) #B, N, 3
        pe = self.pe_embedder(seq_ids) # 2{B, N, 128}
        for blk in self.blocks:
            embedding = auto_grad_checkpoint(blk, embedding, index_embedding, pe)
        
        output1 = self.output_layer(embedding, index_embedding)
        output1 = self.unpatchify(output1).to(torch.float32)[:, -self.configs.output_length:, ...]
        output2 = self.fr_refiner(torch.cat((inputs, output1), dim=1).squeeze(2), index).unsqueeze(2)
        output3 = output1 + output2

        return output1, output2, output3