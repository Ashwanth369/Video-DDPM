import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from einops import repeat


class DDPM(nn.Module):
    def __init__(self, model, time_steps=1000, min_beta=1e-4, max_beta=1e-2, **kwargs):    
        super().__init__(**kwargs)
        
        self.time_steps = time_steps
        self.model = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.beta = torch.linspace(min_beta, max_beta, time_steps, device=self.device)
        self.alpha = 1 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)

    def forward(self, x0, time, eta=None):
        alpha_bar = self.alpha_bar[time]
        if eta is None:
            eta = torch.randn_like(x0)

        mean = alpha_bar.sqrt()
        std = 1 - alpha_bar
        mean = repeat(mean, "b -> b 1 1 1")
        std = repeat(std, "b -> b 1 1 1")

        q_xt = mean * x0 + std.sqrt() * eta

        return q_xt

    def backward(self, xt, time, text_encodings=None, text_mask=None, mask_prob=0.5, generate_sample=False):
        eta_pred = self.model(xt, time, text_encodings, text_mask, mask_prob)
        p_xt = None
        
        if generate_sample:
            beta = self.beta[time]
            alpha = self.alpha[time]
            alpha_bar = self.alpha_bar[time]
            beta = repeat(beta, "b -> b 1 1 1")
            alpha = repeat(alpha, "b -> b 1 1 1")
            alpha_bar = repeat(alpha_bar, "b -> b 1 1 1")

            mean = (xt - (beta/(1-alpha_bar).sqrt()) * eta_pred)/alpha.sqrt()
            eps = torch.randn_like(xt)
            
            p_xt = mean + beta.sqrt() * eps

        return eta_pred, p_xt
    
    def loss(self, x0, eta=None, text_encodings=None, text_mask=None, mask_prob=0.5):
        time = torch.randint(0, self.time_steps, (x0.shape[0], ), dtype=torch.long, device=self.device)
        if eta is None:
            eta = torch.randn_like(x0)
        
        xt = self.forward(x0, time, eta)
        eta_pred, _ = self.backward(xt, time, text_encodings, text_mask, mask_prob)
        loss = F.mse_loss(eta, eta_pred)

        return loss
    
    
if __name__ == "__main__":
    # unet = UNet(image_channels=1)
    ddpm = DDPM()
    x0 = torch.randn((16,1,32,32))
    q_x10 = ddpm.loss(x0)
    # print(q_x10.shape)
    # eta_pred, p_x9 = ddpm.backward(q_x10, 10, generate_sample=True)
    print(q_x10)
    # print(p_x9.shape)
