import torch
import torch.nn as nn


class TimeEmbedding(nn.Module):
    def __init__(self, in_embed_dim, out_embed_dim, theta=10000, **kwargs):    
        super().__init__(**kwargs)

        half_dim = in_embed_dim // 2
        theta = torch.tensor(theta)
        self.embedding = torch.log(theta) / (half_dim - 1)
        self.embedding = torch.exp(torch.arange(half_dim) * -self.embedding)

        self.linear = nn.Sequential(
            nn.Linear(in_embed_dim, out_embed_dim),
            nn.SiLU()
        )

    def timeEmbedding(self, time):
        self.embedding = self.embedding.to(time.device)
        embedding = time[:, None] * self.embedding[None, :]
        embedding = torch.cat((torch.sin(embedding),  torch.cos(embedding)), dim=-1)

        return embedding
    
    def forward(self, time):
        embedding = self.timeEmbedding(time)
        embedding = self.linear(embedding)

        return embedding