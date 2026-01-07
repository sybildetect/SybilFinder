import torch
import torch.nn as nn
import torch.nn.functional as F

class ReverseBiasGate(nn.Module):

    def __init__(self, d_model, num_experts, init_tau = 2.0):
        super().__init__()
        self.num_layers = 4 
        self.linears = nn.ModuleList([
            nn.Linear(d_model, d_model)
            for _ in range(self.num_layers)
        ])
        self.act = nn.GELU()
        self.gamma = nn.Parameter(torch.ones(1, d_model))
        self.fc_layer = nn.Linear(d_model, num_experts, bias=False)
        self.fc_res = nn.Linear(d_model, num_experts, bias=False)
        self.c_sty_in  = nn.Parameter(torch.zeros(num_experts))
        self.c_sem_in  = nn.Parameter(torch.zeros(num_experts))
        self.c_sty_out = nn.Parameter(torch.zeros(num_experts))
        self.c_sem_out = nn.Parameter(torch.zeros(num_experts))
        self.register_buffer("tau", torch.tensor(init_tau))

    def forward(self, H):
        H_layer = H[:, :-1, :]
        H_res = H[:, -1, :]
        Z_res = H_res * self.gamma
        z_list = []
        for i in range(self.num_layers):
            zi = self.act(self.linears[i](H_layer[:, i, :]))  # [B, D']
            z_list.append(zi)
        Z_layer = torch.stack(z_list, dim=1).sum(dim=1)

        G = self.fc_layer(Z_layer)
        B = self.fc_res(Z_res)

        g_sty_in = (B + G) / self.num_layers + self.c_sty_in
        g_sem_in = (B - G) / self.num_layers + self.c_sem_in
        g_in = torch.stack([g_sty_in, g_sem_in], dim=-1)
        w_in = F.softmax(g_in, dim=-1)
        alpha_in = w_in[..., 0]
        beta_in  = w_in[..., 1]

        g_sty_out = (B + G) / self.tau + self.c_sty_out
        g_sem_out = (B - G) / self.tau + self.c_sem_out
        alpha_out = F.softmax(g_sty_out, dim=-1)
        beta_out  = F.softmax(g_sem_out, dim=-1)

        return alpha_in, beta_in, alpha_out, beta_out


class SimilarityGate(nn.Module):

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(3, 1)

    def forward(self, s_a, s_b):
        ratio = s_a / (s_a + s_b + 1e-8)
        x = torch.cat([s_a, s_b, ratio], dim=1)
        return torch.sigmoid(self.fc(x))
