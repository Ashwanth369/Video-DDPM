import torch
import torch.nn as nn
from utils import ConvBlock, MidBlock, DownSample, UpSample, LayerBlock


class ImageAutoEncoder(nn.Module):
    def __init__(self, image_channels=3, dim=32, channel_mult=(1,2,4), attn_layers=[False,False,True],
                 encode_dim=None, groups=32, kernel_size=(3,3), padding=(1,1), dropout=0.0,
                 n_heads=8, head_dim=32, **kwargs):
        super().__init__(**kwargs)
        
        embed_dim = None
        
        n_resolutions = len(channel_mult)
        out_channels = [dim * mult for mult in channel_mult]
        in_channels = [dim] + out_channels[:-1]
        encode_dim = out_channels[-1] if encode_dim is None else encode_dim
        
        encoder_layers = [nn.Conv2d(image_channels, dim, kernel_size=(3,3), padding=(1,1))]
        for idx in range(n_resolutions):
            encoder_layers.append(LayerBlock(groups, in_channels[idx], out_channels[idx], embed_dim, kernel_size, 
                                             padding, attn_layers[idx], dropout, n_heads, head_dim))
            if idx < n_resolutions-1:
                encoder_layers.append(DownSample(out_channels[idx]))
        
        encoder_layers.append(MidBlock(groups, out_channels[-1], embed_dim, kernel_size, padding, 
                                       dropout, n_heads, head_dim, **kwargs))
        encoder_layers.append(ConvBlock(groups, out_channels[-1], encode_dim*2, kernel_size, padding, dropout))
        self.encoder_blocks = nn.Sequential(*encoder_layers)

        decoder_layers = [nn.Conv2d(encode_dim, out_channels[-1], kernel_size=(3,3), padding=(1,1))]
        decoder_layers.append(MidBlock(groups, out_channels[-1], embed_dim, kernel_size, padding, 
                                       dropout, n_heads, head_dim, **kwargs))
        for idx in range(n_resolutions-1, -1, -1):
            decoder_layers.append(LayerBlock(groups, out_channels[idx], in_channels[idx], embed_dim, kernel_size, 
                                             padding, attn_layers[idx], dropout, n_heads, head_dim))
            if idx > 0:
                decoder_layers.append(UpSample(in_channels[idx]))

        decoder_layers.append(ConvBlock(groups, in_channels[0], image_channels, kernel_size, padding, dropout))
        self.decoder_blocks = nn.Sequential(*decoder_layers)

    @property
    def downsample_factor(self):
        return 2 ** (len(self.down_blocks) - 1)

    def forward(self, xt):
        encodings = self.getEncoding(xt)
        mean, log_var = torch.chunk(encodings, 2, dim=1)
        latent = self.getLatent(mean, log_var)
        out = self.getDecoding(latent)
        
        return out, mean, log_var
    
    def getLatent(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        return mean + std * torch.randn_like(std)


    def getEncoding(self, xt):
        encodings = self.encoder_blocks(xt)
        return encodings

    def getDecoding(self, latent):
        out = self.decoder_blocks(latent)
        return out
    

def KLDivergence(mean, log_var):
    return -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

def getImageEncodings(images, model=None):
    with torch.no_grad():
        image_encodings = model.getEncoding(images)

    return image_encodings

def getImageDecodings(image_encodings, model=None):
    with torch.no_grad():
        image_decodings = model.getDecoding(image_encodings)

    return image_decodings


if __name__ == "__main__":
    imgAE = ImageAutoEncoder(image_channels=1)
    inp = torch.randn((4,1,32,32))
    print(imgAE(inp).shape)