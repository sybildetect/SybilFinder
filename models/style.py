import torch
import torch.nn as nn
import torch.nn.functional as F

class StyleSimilarity(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.fc = nn.Linear(d_model, d_model // 2, bias=False)

    def forward(self, T1, T2, mask1, mask2):
        v1 = self._style_vector(T1, mask1)
        v2 = self._style_vector(T2, mask2)
        sim = F.cosine_similarity(v1, v2, dim=1, eps=1e-8)
        sim = (sim + 1.0) / 2.0
        return sim.unsqueeze(1)

    def _style_vector(self, T, mask):
        T = self.fc(T)
        d = T.size(-1)
        A = torch.matmul(T, T.transpose(1, 2)) / (d ** 0.5)
        A = A.masked_fill(mask.unsqueeze(1) == 0, -1e9)
        A = F.softmax(A, dim=-1)
        H = torch.matmul(A, T)
        H = H * mask.unsqueeze(-1)
        v = H.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1)
        return v
