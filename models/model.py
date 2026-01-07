import torch
import torch.nn as nn
import torch.nn.functional as F

from models.experts import TextExpert, ImageExpert
from models.style import StyleSimilarity
from models.semantic import SemanticSimilarity
from models.gates import ReverseBiasGate, SimilarityGate

class SiameseNetwork(nn.Module):
    def __init__(
        self,
        embedding_txt,
        embedding_img,
        hidden_size,
        experts_num,

    ):
        super().__init__()

        self.num_experts = experts_num
        self.hidden_size = hidden_size

        self.stream_text = torch.cuda.Stream()
        self.stream_image = torch.cuda.Stream()

        self.text_experts = nn.ModuleList([
            TextExpert(embedding_txt, hidden_size)
            for _ in range(experts_num)
        ])

        self.image_experts = nn.ModuleList([
            ImageExpert(embedding_img, hidden_size)
            for _ in range(experts_num)
        ])

        self.text_moe_gate = ReverseBiasGate(embedding_txt, experts_num)
        self.text_semantic_branch = SemanticSimilarity()
        self.text_style_branch = StyleSimilarity(hidden_size)

        self.image_moe_gate = ReverseBiasGate(embedding_img, experts_num)
        self.image_semantic_branch = SemanticSimilarity()
        self.image_style_branch = StyleSimilarity(hidden_size)

        self.text_sim_gate  = SimilarityGate()
        self.image_sim_gate = SimilarityGate()
        self.modal_sim_gate = SimilarityGate()

    def forward(self, A_x, gate_A_x, A_img, gate_A_img, B_x, gate_B_x, B_img, gate_B_img, tau=2.0):

        valid_A = A_img.abs().sum(dim=(1, 2, 3)) > 0   # [B]
        valid_B = B_img.abs().sum(dim=(1, 2, 3)) > 0   # [B]
        valid_img = (valid_A & valid_B).float()

        with torch.cuda.stream(self.stream_text):
            s_text = \
                self._modality_forward(
                    A_x, B_x, gate_A_x, gate_B_x,
                    experts=self.text_experts,
                    moe_gate=self.text_moe_gate,
                    semantic_branch=self.text_semantic_branch,
                    style_branch=self.text_style_branch,
                    sim_gate=self.text_sim_gate
                )

        with torch.cuda.stream(self.stream_image):
            s_image = \
                self._modality_forward(
                    A_img, B_img, gate_A_img, gate_B_img,
                    experts=self.image_experts,
                    moe_gate=self.image_moe_gate,
                    semantic_branch=self.image_semantic_branch,
                    style_branch=self.image_style_branch,
                    sim_gate=self.image_sim_gate
                )

        torch.cuda.current_stream().wait_stream(self.stream_text)
        torch.cuda.current_stream().wait_stream(self.stream_image)

        gamma = self.modal_sim_gate(s_text, s_image)
        gamma = gamma * valid_img.view(-1, 1) 
        s_image = s_image * valid_img.view(-1, 1) 

        alpha = 1.0 / (1.0 + gamma)
        beta  = gamma / (1.0 + gamma)

        s = alpha * s_text + beta * s_image

        return {
            "similarity": s.squeeze(1)
        }

    def _modality_forward(
        self,
        A_seq,
        B_seq,
        A_gate,
        B_gate,
        experts,
        moe_gate,
        semantic_branch,
        style_branch,
        sim_gate
    ):

        A_mask = self.make_padding_mask(A_seq)
        B_mask = self.make_padding_mask(B_seq)

        A_E, alpha_A, beta_A = self.moe_encode(A_seq, A_gate, A_mask, experts, moe_gate)
        B_E, alpha_B, beta_B = self.moe_encode(B_seq, B_gate, B_mask, experts, moe_gate)

        A_style = (alpha_A.unsqueeze(-1).unsqueeze(-1) * A_E).sum(dim=1)
        B_style = (alpha_B.unsqueeze(-1).unsqueeze(-1) * B_E).sum(dim=1)
        s_style = style_branch(A_style, B_style, A_mask, B_mask)

        sim_mats = self.pairwise_cos(A_E, B_E)  # [B, T, T]
        w = 0.5 * (beta_A + beta_B)
        sim_mat = (w.unsqueeze(-1).unsqueeze(-1) * sim_mats).sum(dim=1)
        s_sem = semantic_branch(sim_mat, A_mask, B_mask)

        lam = sim_gate(s_sem, s_style)

        s = lam * s_sem + (1.0 - lam) * s_style

        return s

    def set_moe_temperature(self, tau: float):
        self.text_moe_gate.tau.fill_(tau)
        self.image_moe_gate.tau.fill_(tau)

    def make_padding_mask(self, seq):
        mask = seq.abs().sum(dim=(-1, -2)) > 0
        return mask.float()
    
    def moe_encode(self, X_seq, X_gate, X_mask, experts, moe_gate):
        X_Es = []
        alpha_in, beta_in, alpha_out, beta_out = moe_gate(X_gate)
        X_fused = self.gate_fuse(X_seq, alpha_in, beta_in)
        for l, expert in enumerate(experts):
            X_Es.append(expert.forward(X_fused[:, l], X_mask))
        X_E = torch.stack(X_Es, dim=1)
        return X_E, alpha_out, beta_out

    def gate_fuse(self, seq, alpha, beta):
        x_low = seq[:, :, 0, :].unsqueeze(1)          # [B, 1, T, H]
        x_high = seq[:, :, 1, :].unsqueeze(1)
        a = alpha.unsqueeze(-1).unsqueeze(-1)  # [B, L, 1, 1]
        b = beta.unsqueeze(-1).unsqueeze(-1)
        return a * x_low + b * x_high

    def pairwise_cos(self, A, B):
        A = F.normalize(A, dim=-1, eps=1e-8)
        B = F.normalize(B, dim=-1, eps=1e-8)
        cos = torch.matmul(A, B.transpose(-1, -2))
        return (cos + 1.0) / 2.0