# models/semantic.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class SemanticSimilarity(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)

    def forward(self, text_sim, mask1, mask2):
        B, T, _ = text_sim.size()
        mask = mask1.unsqueeze(2) * mask2.unsqueeze(1)
        x = text_sim.unsqueeze(1)
        A = torch.sigmoid(self.conv(x))
        A = A.view(B, -1)
        A = A.masked_fill(mask.view(B, -1) == 0, -1e9)
        pi = F.softmax(A, dim=1)              
        s = torch.sum(pi * text_sim.view(B, -1), dim=1, keepdim=True)
        return s
