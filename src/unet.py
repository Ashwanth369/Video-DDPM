import torch
import torch.nn as nn
from einops import rearrange
from utils import LayerBlock, CrossAttentionBlock, MidBlock, DownSample, UpSample, ConvBlock

from time_embedder import TimeEmbedding


class UNet(nn.Module):
    def __init__(self, input_channels=256, embed_dim=64, cond_dim=128, channel_mult=(1,2,4), text_embed_dim=512,
                 max_seq_len=8, time_embed_dim=None, groups=8, kernel_size=(3,3), padding=(1,1), dropout=0.0,
                 n_heads=4, head_dim=32, attn_layers=[False,False,True], num_time_tokens=2, **kwargs):
        super().__init__(**kwargs)
        
        n_resolutions = len(channel_mult)
        time_embed_dim = embed_dim // 2 if time_embed_dim is None else time_embed_dim
        self.num_time_tokens = num_time_tokens
        self.max_seq_len = max_seq_len
        
        self.time_hiddens = TimeEmbedding(time_embed_dim, embed_dim, **kwargs)
        self.time_to_cond = nn.Linear(embed_dim, cond_dim * num_time_tokens)
        self.time_to_embed = nn.Linear(embed_dim, embed_dim)

        self.text_to_cond = nn.Linear(text_embed_dim, cond_dim)
        self.null_text_cond = nn.Parameter(torch.randn(1, max_seq_len, cond_dim))
        self.text_to_embed = nn.Sequential(
            nn.LayerNorm(cond_dim),
            nn.Linear(cond_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.null_text_embed = nn.Parameter(torch.randn(1, embed_dim))

        conv_dim = input_channels if input_channels > embed_dim else embed_dim
        self.init_conv = nn.Conv2d(input_channels, conv_dim, kernel_size=(3,3), padding=(1,1))

        out_channels = [conv_dim * mult for mult in channel_mult]
        in_channels = [conv_dim] + out_channels[:-1]

        downs = list()
        for idx in range(n_resolutions):
            downs.append(CrossAttentionBlock(groups, in_channels[idx], cond_dim))
            downs.append(LayerBlock(groups, in_channels[idx], out_channels[idx], embed_dim, kernel_size, 
                                    padding, False, dropout, n_heads, head_dim, **kwargs))
            downs.append(LayerBlock(groups, out_channels[idx], out_channels[idx], embed_dim, kernel_size, 
                                    padding, attn_layers[idx], dropout, n_heads, head_dim, **kwargs))
            if idx < n_resolutions-1:
                downs.append(DownSample(out_channels[idx]))

        self.down_blocks = nn.ModuleList(downs)
        
        self.middle_block = MidBlock(groups, out_channels[-1], embed_dim, kernel_size, padding, 
                                     dropout, n_heads, head_dim, **kwargs)
        
        ups = list()
        for idx in range(n_resolutions-1, -1, -1):
            ups.append(CrossAttentionBlock(groups, out_channels[idx], cond_dim))
            ups.append(LayerBlock(groups, 2*out_channels[idx], out_channels[idx], embed_dim,
                                   kernel_size, padding, False, dropout, n_heads, head_dim, **kwargs))
            ups.append(LayerBlock(groups, 2*out_channels[idx], in_channels[idx], embed_dim,
                                   kernel_size, padding, attn_layers[idx], dropout, n_heads, head_dim, **kwargs))
            if idx > 0:
                ups.append(UpSample(in_channels[idx]))

        self.up_blocks = nn.ModuleList(ups)

        self.final_conv = ConvBlock(groups, in_channels[0], input_channels, kernel_size, padding, dropout)

    @property
    def downsample_factor(self):
        return 2 ** (len(self.down_blocks) - 1)

    def forward(self, xt, time, text_encodings=None, text_mask=None, mask_prob=0.5):
        xt = self.init_conv(xt)
        
        time = self.time_hiddens(time)
        embed = self.time_to_embed(time)
        cond = self.time_to_cond(time)
        cond = rearrange(cond, 'b (r d) -> b r d', r=self.num_time_tokens)
        cond_mask = torch.ones((xt.shape[0], self.num_time_tokens), device=xt.device).bool()
        
        if text_encodings is not None:
            rand_mask = torch.zeros((xt.shape[0], ), device=xt.device).uniform_(0,1) < mask_prob
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
        
        skips = list()
        for block in self.down_blocks:
            if isinstance(block, CrossAttentionBlock):
                xt = block(xt, cond, cond_mask)
            elif isinstance(block, LayerBlock):
                xt = block(xt, embed)
                skips.append(xt)
            else:
                xt = block(xt)

        xt = self.middle_block(xt, embed)

        for block in self.up_blocks:
            if isinstance(block, CrossAttentionBlock):
                xt = block(xt, cond, cond_mask)
            elif isinstance(block, LayerBlock):
                xt = torch.cat((xt, skips.pop()), dim=1)
                xt = block(xt, embed)
            else:
                xt = block(xt)

        out = self.final_conv(xt)
        return out


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x0 = torch.randn((16,256,8,8)).to(device)
    time = x0.new_full((16,), 1, dtype=torch.long).to(device)
    unet = UNet().to(device)
    print(unet(x0, time).shape)