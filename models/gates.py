import torch
import torch.nn as nn
import torch.nn.functional as F

class ReverseBiasGate(nn.Module):

    def __init__(self, num_layers, d_model, num_experts, init_tau = 2.0):
        super().__init__()
        self.linears = nn.ModuleList([
            nn.Linear(d_model, d_model // num_layers)
            for _ in range(num_layers)
        ])
        self.act = nn.GELU()
        self.gamma = nn.Parameter(torch.ones(1, d_model))
        self.fc = nn.Linear(d_model * 2, num_experts, bias=False)
        self.c_sty = nn.Parameter(torch.zeros(num_experts))
        self.c_sem = nn.Parameter(torch.zeros(num_experts))
        self.register_buffer("tau", torch.tensor(init_tau))

    def forward(self, H):
        H_res = H[:, -1, :]
        H_layer = H[:, :-1, :]
        Z_res = H_res * self.gamma
        z_list = []
        for i in range(H_layer.size(1)):
            zi = self.act(self.linears[i](H_layer[:, i, :]))  # [B, D']
            z_list.append(zi)
        Z_layer = torch.stack(z_list, dim=1)  # [B, T-1, D']
        Z_layer = Z_layer.reshape(Z_layer.size(0), -1)
        Z = torch.cat([Z_layer, Z_res], dim=1)  

        G = self.fc(Z)             # [B, L]

        g_sty = G + self.c_sty
        g_sem = -G + self.c_sem

        alpha = F.softmax(g_sty / self.tau, dim=-1)
        beta  = F.softmax(g_sem / self.tau, dim=-1)

        return alpha, beta


class SimilarityGate(nn.Module):

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(3, 1)

    def forward(self, s_a, s_b):
        ratio = s_a / (s_a + s_b + 1e-8)
        x = torch.cat([s_a, s_b, ratio], dim=1)
        return torch.sigmoid(self.fc(x))
