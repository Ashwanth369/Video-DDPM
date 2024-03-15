import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from einops import rearrange, repeat

from time_embedder import TimeEmbedding
from text_encoder import getTextEncoding


class LinearBlock(nn.Module):
    def __init__(self, in_channels, out_channels, embed_dim, dropout=0.2):
        super().__init__()

        def layerBlock(in_c, out_c):
            layer = nn.Sequential(
                nn.LayerNorm(in_c),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(in_c, out_c)
            )
            return layer

        self.linear1 = layerBlock(in_channels, out_channels)
        self.linear2 = layerBlock(out_channels, out_channels)

        self.embed_to_channel = nn.Linear(embed_dim, out_channels)

    def forward(self, xt, embed):
        out = self.linear1(xt)
        out += self.embed_to_channel(embed)[:, None, None, :]
        out = self.linear2(out)

        return out


class CrossAttentionBlock(nn.Module):
    def __init__(self, n_channels, context_dim, n_heads=4, head_dim=None, **kwargs):    
        super().__init__(**kwargs)
        
        if head_dim is None:
            head_dim = n_channels

        self.n_channels = n_channels
        self.heads = n_heads
        self.hidden_dim = n_heads * head_dim
        self.scale = head_dim ** -0.5
        
        self.norm = nn.LayerNorm(n_channels)
        self.context_norm = nn.LayerNorm(context_dim)
        
        self.to_q = nn.Linear(n_channels, self.hidden_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, self.hidden_dim * 2, bias=False)
        self.null_kv = nn.Parameter(torch.randn(2, head_dim))

        self.output = nn.Linear(self.hidden_dim, n_channels)

    def forward(self, xt, context, mask=None, return_residual=True):
        xt = self.norm(xt)
        context = self.context_norm(context)

        q = self.to_q(xt)
        q = rearrange(q, "b x y (h c) -> b h (x y) c", h=self.heads)
        
        k, v = self.to_kv(context).chunk(2, dim=-1)
        k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), (k,v))
        nk, nv = map(lambda t: repeat(t, "d -> b h 1 d", b=xt.shape[0], h=self.heads), self.null_kv)
        k = torch.cat((nk, k), dim = -2)
        v = torch.cat((nv, v), dim = -2)
        
        similarity = torch.einsum("b h i d, b h j d -> b h i j", q, k) * self.scale
        if mask is not None:
            max_neg_value = -torch.finfo(similarity.dtype).max
            mask = F.pad(mask, (1,0), value=True)
            mask = rearrange(mask, 'b j -> b 1 1 j')
            similarity = similarity.masked_fill(~mask, max_neg_value)
        
        attention = similarity.softmax(dim=-1, dtype=torch.float32)
        out = torch.einsum("b h i j, b h j d -> b h i d", attention, v)
        out = rearrange(out, "b h (x y) c -> b x y (h c)", x=xt.shape[1], y=xt.shape[2])
        out = self.output(out)
        if return_residual:
            out += xt
        
        return out


class LDM(nn.Module):
    def __init__(self, input_channels=512, dim=64, channel_mult=(2,4,2), text_embed_dim=512, max_seq_len=8,
                 time_embed_dim=None,dropout=0.2, n_heads=4, head_dim=32, num_time_tokens=2, **kwargs):
        super().__init__(**kwargs)
        
        # n_resolutions = len(channel_mult)
        embed_dim = dim * 4
        time_embed_dim = embed_dim // 4 if time_embed_dim is None else time_embed_dim
        self.num_time_tokens = num_time_tokens
        self.max_seq_len = max_seq_len
        
        self.time_hiddens = TimeEmbedding(time_embed_dim, embed_dim, **kwargs)
        self.time_to_cond = nn.Linear(embed_dim, dim * num_time_tokens)
        self.time_to_embed = nn.Linear(embed_dim, embed_dim)

        self.text_to_cond = nn.Linear(text_embed_dim, dim)
        self.null_text_cond = nn.Parameter(torch.randn(1, max_seq_len, dim))
        self.text_to_embed = nn.Sequential(
                                nn.LayerNorm(dim),
                                nn.Linear(dim, embed_dim),
                                nn.SiLU(),
                                nn.Linear(embed_dim, embed_dim),
                            )
        self.null_text_embed = nn.Parameter(torch.randn(1, embed_dim))
        self.init_layer = nn.Linear(input_channels, input_channels)

        linear_layers = []
        prev_mult = 1
        for mult in channel_mult:
            linear_layers.append(CrossAttentionBlock(input_channels * prev_mult, dim, n_heads, head_dim))
            linear_layers.append(LinearBlock(input_channels * prev_mult, input_channels * mult, embed_dim, dropout))
            prev_mult = mult

        self.autoencoder = nn.ModuleList(linear_layers)

        self.final_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(input_channels * prev_mult, input_channels)
        )


    def forward(self, xt, time, text_encodings=None, text_mask=None):
        xt = self.init_layer(xt)
        
        time = self.time_hiddens(time)
        embed = self.time_to_embed(time)
        cond = self.time_to_cond(time)
        cond = rearrange(cond, 'b (r d) -> b r d', r=self.num_time_tokens)
        cond_mask = torch.ones((xt.shape[0], self.num_time_tokens), device=xt.device).bool()
        
        if text_encodings is not None:
            rand_mask = torch.zeros((xt.shape[0], ), device=xt.device).uniform_(0,1) < 0.5
            text_cond_mask = rearrange(rand_mask, "b -> b 1 1")
            text_embed_mask = rearrange(rand_mask, "b -> b 1")
            
            if text_mask is not None:
                text_cond_mask = text_cond_mask & rearrange(text_mask, 'b n -> b n 1')

            text_cond = self.text_to_cond(text_encodings)
            text_cond = torch.where(text_cond_mask, text_cond, self.null_text_cond)
            mean_text_cond = text_cond.mean(dim=-2)

            text_embed = self.text_to_embed(mean_text_cond)
            text_embed = torch.where(text_embed_mask, text_embed, self.null_text_embed)

            embed += text_embed
            cond = torch.cat((cond, text_cond), dim=-2)
            cond_mask = torch.cat((cond_mask, text_mask), dim=-1)
        
        for block in self.autoencoder:
            if isinstance(block, CrossAttentionBlock):
                xt = block(xt, cond, cond_mask)
            else:
                xt = block(xt, embed)

        out = self.final_layer(xt)
        return out


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x0 = torch.randn((16,512,4,4)).to(device)
    time = x0.new_full((16,), 1, dtype=torch.long)
    x0 = x0.permute(0,2,3,1)
    print(x0.shape, time.shape)
    text_list = ["one"]*16
    text_encodings, text_mask = getTextEncoding(text_list)
    ldm = LDM(input_channels=512).to(device)
    print(ldm(x0, time, text_encodings, text_mask).shape)