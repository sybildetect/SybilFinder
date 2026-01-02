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
        embedding_gate,
        hidden_size,
        experts_num,

    ):
        super().__init__()

        self.num_experts = experts_num
        self.hidden_size = hidden_size

        self.text_experts = nn.ModuleList([
            TextExpert(embedding_txt * 2, hidden_size)
            for _ in range(experts_num)
        ])

        self.image_experts = nn.ModuleList([
            ImageExpert(embedding_img * 2, hidden_size)
            for _ in range(experts_num)
        ])

        self.text_moe_gate = ReverseBiasGate(embedding_gate, experts_num)
        self.image_moe_gate = ReverseBiasGate(embedding_gate, experts_num)
        self.text_semantic_branch = SemanticSimilarity()
        self.text_style_branch = StyleSimilarity(hidden_size * 2)

        self.image_semantic_branch = SemanticSimilarity()
        self.image_style_branch = StyleSimilarity(hidden_size * 2)

        self.text_sim_gate  = SimilarityGate()
        self.image_sim_gate = SimilarityGate()
        self.modal_sim_gate = SimilarityGate()

    def forward(self, A_x, Ap_x, A_img, Ap_img, B_x, Bp_x, B_img, Bp_img, tau=2.0):

        valid_A = A_img.abs().sum(dim=(1, 2)) > 0   # [B]
        valid_B = B_img.abs().sum(dim=(1, 2)) > 0   # [B]
        valid_img = valid_A & valid_B
        

        stream_text = torch.cuda.Stream()
        stream_image = torch.cuda.Stream()

        with torch.cuda.stream(stream_text):
            s_text, lamda, gate_sty, gate_sem, s_sem, s_style = \
                self._modality_forward(
                    A_x, B_x, Ap_x, Bp_x,
                    experts=self.text_experts,
                    moe_gate=self.text_moe_gate,
                    semantic_branch=self.text_semantic_branch,
                    style_branch=self.text_style_branch,
                    sim_gate=self.text_sim_gate
                )

        with torch.cuda.stream(stream_image):
            s_image, *_ = \
                self._modality_forward(
                    A_img, B_img, Ap_img, Bp_img,
                    experts=self.image_experts,
                    moe_gate=self.image_moe_gate,
                    semantic_branch=self.image_semantic_branch,
                    style_branch=self.image_style_branch,
                    sim_gate=self.image_sim_gate
                )
                

        torch.cuda.synchronize()

        gamma = self.modal_sim_gate(s_text, s_image)
        gamma = gamma * valid_img.view(-1, 1) 
        s_image = s_image * valid_img.view(-1, 1) 

        alpha = 1.0 / (1.0 + gamma)
        beta  = gamma / (1.0 + gamma)

        s = alpha * s_text + beta * s_image

        return {
            "similarity": s.squeeze(1),
            "sim_gate": lamda.squeeze(1),
            "sem_sim": s_sem.squeeze(1),
            "sty_sim": s_style.squeeze(1),
            "gate_sty": gate_sty, "gate_sem": gate_sem,
            "text_sim": (s_text, lamda, s_sem, s_style),
            "image_sim": s_image,
            "modality_gate": alpha     
        }

    def _modality_forward(
        self,
        A_seq,
        B_seq,
        A_p,
        B_p,
        experts,
        moe_gate,
        semantic_branch,
        style_branch,
        sim_gate
    ):

        A_mask = self.make_padding_mask(A_seq)
        B_mask = self.make_padding_mask(B_seq)

        A_Es, B_Es = [], []

        stream_A = torch.cuda.Stream()
        stream_B = torch.cuda.Stream()

        with torch.cuda.stream(stream_A):
            alpha_A, beta_A = moe_gate(A_p)
            for expert in experts:
                A_E = expert.encode(A_seq, A_mask)   # [B,T,2H]
                A_Es.append(A_E)

        with torch.cuda.stream(stream_B):
            alpha_B, beta_B = moe_gate(B_p)
            for expert in experts:
                B_E = expert.encode(B_seq, B_mask)
                B_Es.append(B_E)

        torch.cuda.synchronize()

        A_style = self._moe_weighted_sum(A_Es, alpha_A)
        B_style = self._moe_weighted_sum(B_Es, alpha_B)
        s_style = style_branch(A_style, B_style, A_mask, B_mask)


        sim_mat = 0.0
        for l in range(self.num_experts):
            S_l = self._pairwise_cos(A_Es[l], B_Es[l])  # [B, T, T]
            w = 0.5 * (beta_A[:, l] + beta_B[:, l]).view(-1, 1, 1)
            sim_mat = sim_mat + w * S_l

        s_sem = semantic_branch(sim_mat, A_mask, B_mask)


        lam = sim_gate(s_sem, s_style)
        return lam * s_sem + (1.0 - lam) * s_style, lam, (alpha_A + alpha_B) / 2, (beta_A + beta_B) / 2, s_sem, s_style

    def _moe_weighted_sum(self, E_list, weights):
        out = 0.0
        for l, E in enumerate(E_list):
            w = weights[:, l].view(-1, 1, 1)
            out = out + w * E
        return out

    def _pairwise_cos(self, A, B):
        A = F.normalize(A, dim=-1)
        B = F.normalize(B, dim=-1)
        cos = torch.matmul(A, B.transpose(1, 2))
        return (cos + 1.0) / 2.0

    def make_padding_mask(self, seq):
        mask = (seq.abs().sum(dim=-1) > 0)
        return mask.float()
    
    def set_moe_temperature(self, tau: float):
        self.text_moe_gate.tau.fill_(tau)
        self.image_moe_gate.tau.fill_(tau)