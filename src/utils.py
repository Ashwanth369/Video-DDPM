import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat


class ConvBlock(nn.Module):
    def __init__(self, n_groups, in_channels, out_channels, kernel_size=(3,3), padding=(1,1), dropout=0.2):
        super().__init__()

        self.conv_block = nn.Sequential(
            nn.GroupNorm(n_groups, in_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        )

    def forward(self, xt):
        return self.conv_block(xt)


class ResidualBlock(nn.Module):
    def __init__(self, groups, in_channels, out_channels, embed_dim=None,
                 kernel_size=(3,3), padding=(1,1), dropout=0.0, **kwargs):
        super().__init__(**kwargs)

        self.adjust_input = nn.Conv2d(in_channels, out_channels, kernel_size=(1,1)) \
                                     if in_channels != out_channels else nn.Identity()

        self.conv_block1 = ConvBlock(groups, in_channels, out_channels, kernel_size, padding, dropout)
        self.conv_block2 = ConvBlock(groups, out_channels, out_channels, kernel_size, padding, dropout)

        if embed_dim is not None:
            self.embed_to_channel = nn.Sequential(
                nn.SiLU(),
                nn.Linear(embed_dim, out_channels)
            )

    def forward(self, xt, embed=None):
        out = self.conv_block1(xt)
        if embed is not None:
            out += self.embed_to_channel(embed)[:, :, None, None]

        out = self.conv_block2(out)
        out += self.adjust_input(xt)

        return out


class AttentionBlock(nn.Module):
    def __init__(self, n_groups, n_channels, n_heads=4, head_dim=None, **kwargs):
        super().__init__(**kwargs)

        if head_dim is None:
            head_dim = n_channels

        self.n_channels = n_channels
        self.heads = n_heads
        self.hidden_dim = n_heads * head_dim
        self.scale = head_dim ** -0.5

        self.norm = nn.GroupNorm(n_groups, n_channels)

        self.to_qkv = nn.Conv2d(n_channels, self.hidden_dim * 3, kernel_size=(1,1), bias=False)
        self.null_kv = nn.Parameter(torch.randn(2, head_dim))

        self.output = nn.Conv2d(self.hidden_dim, n_channels, kernel_size=(1,1))

    def forward(self, xt, return_residual=True):
        xt = self.norm(xt)

        qkv = self.to_qkv(xt).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, "b (h c) x y -> b h (x y) c", h=self.heads), qkv)

        nk, nv = map(lambda t: repeat(t, "d -> b h 1 d", b=xt.shape[0], h=self.heads), self.null_kv)
        k = torch.cat((nk, k), dim = -2)
        v = torch.cat((nv, v), dim = -2)

        similarity = torch.einsum("b h i d, b h j d -> b h i j", q, k) * self.scale

        attention = similarity.softmax(dim=-1, dtype=torch.float32)
        out = torch.einsum("b h i j, b h j d -> b h i d", attention, v)

        out = rearrange(out, "b h (x y) c -> b (h c) x y", x=xt.shape[2], y=xt.shape[3])
        out = self.output(out)
        if return_residual:
            out += xt

        return out


class CrossAttentionBlock(nn.Module):
    def __init__(self, n_groups, n_channels, context_dim, n_heads=4, head_dim=None, **kwargs):
        super().__init__(**kwargs)

        if head_dim is None:
            head_dim = n_channels

        self.n_channels = n_channels
        self.heads = n_heads
        self.hidden_dim = n_heads * head_dim
        self.scale = head_dim ** -0.5

        self.norm = nn.GroupNorm(n_groups, n_channels)
        self.context_norm = nn.GroupNorm(n_groups, context_dim)

        self.to_q = nn.Conv2d(n_channels, self.hidden_dim, kernel_size=(1,1), bias=False)
        self.to_kv = nn.Linear(context_dim, self.hidden_dim * 2, bias=False)
        self.null_kv = nn.Parameter(torch.randn(2, head_dim))

        self.output = nn.Conv2d(self.hidden_dim, n_channels, kernel_size=(1,1))

    def forward(self, xt, context, mask=None, return_residual=True):
        xt = self.norm(xt)
        context = torch.permute(context, (0, 2, 1))
        context = self.context_norm(context)
        context = torch.permute(context, (0, 2, 1))

        q = self.to_q(xt)
        q = rearrange(q, "b (h c) x y -> b h (x y) c", h=self.heads)

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

        out = rearrange(out, "b h (x y) c -> b (h c) x y", x=xt.shape[2], y=xt.shape[3])
        out = self.output(out)
        if return_residual:
            out += xt

        return out


class LayerBlock(nn.Module):
    def __init__(self, groups, in_channels, out_channels, embed_dim=None, kernel_size=(3,3),
                 padding=(1,1), attention=False, dropout=0.0, n_heads=4, head_dim=32, **kwargs):
        super().__init__(**kwargs)

        self.res_block1 = ResidualBlock(groups, in_channels, out_channels, embed_dim,
                                       kernel_size, padding, dropout, **kwargs)
        self.res_block2 = ResidualBlock(groups, out_channels, out_channels, embed_dim,
                                       kernel_size, padding, dropout, **kwargs)
        self.attention_block = AttentionBlock(groups, out_channels, n_heads, head_dim,
                                              **kwargs) if attention else nn.Identity()

    def forward(self, xt, embed=None):
        out = self.res_block1(xt, embed)
        out = self.res_block2(out, embed)
        out = self.attention_block(out)

        return out


class MidBlock(nn.Module):
    def __init__(self, groups, n_channels, embed_dim=None, kernel_size=(3,3), padding=(1,1),
                 dropout=0.0, n_heads=4, head_dim=32, **kwargs):
        super().__init__(**kwargs)

        self.res_block1 = ResidualBlock(groups, n_channels, n_channels, embed_dim,
                                        kernel_size, padding, dropout, **kwargs)
        self.attention_block = AttentionBlock(groups, n_channels, n_heads, head_dim, **kwargs)
        self.res_block2 = ResidualBlock(groups, n_channels, n_channels, embed_dim,
                                        kernel_size, padding, dropout, **kwargs)


    def forward(self, xt, embed=None):
        out = self.res_block1(xt, embed)
        out = self.attention_block(out)
        out = self.res_block2(xt, embed)

        return out
    

class UpSample(nn.Module):
    def __init__(self, n_channels):
        super().__init__()
        self.up_sample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(n_channels, n_channels, kernel_size=(3,3), padding=(1,1))
        )

    def forward(self, xt):
        return self.up_sample(xt)

class DownSample(nn.Module):
    def __init__(self, n_channels):
        super().__init__()
        self.down_sample = nn.Conv2d(n_channels, n_channels, kernel_size=(3,3), stride=(2,2), padding=(1,1))
    
    def forward(self, xt):
        return self.down_sample(xt)
    

def gaussianDistribution(encodings):
    mean, log_var = torch.chunk(encodings, 2, dim=1)
    log_var = torch.clamp(log_var, -30.0, 20.0)
    std = torch.exp(0.5 * log_var)

    return mean + std * torch.randn_like(std)