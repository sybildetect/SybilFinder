import torch
import torch.nn as nn
import torch.nn.functional as F

class SemanticSimilarity(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)

    def forward(self, text_sim, mask1, mask2):
        mask = mask1.unsqueeze(2) * mask2.unsqueeze(1)
        pi = self._spatial_attn(text_sim, mask)
        pi_t = self._spatial_attn(text_sim.transpose(1,2), mask.transpose(1,2))
        pi = (pi + pi_t.transpose(1, 2)) / 2.0
        s = torch.sum(pi * text_sim, dim=(1, 2), keepdim=True)
        return s.squeeze(-1)
    
    def _spatial_attn(self, text_sim, mask):
        B, T1, T2 = text_sim.size()
        x = text_sim.unsqueeze(1)
        A = torch.sigmoid(self.conv(x))          # [B,1,5,5]
        A = A.reshape(B, -1)
        A = A.masked_fill(mask.reshape(B, -1) == 0, -1e9)
        pi = F.softmax(A, dim=1)
        pi = pi.view(B, T1, T2)
        return pi