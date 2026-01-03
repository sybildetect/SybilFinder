import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class TextExpert(nn.Module):

    def __init__(self, embedding_size, hidden_size):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(embedding_size, embedding_size // 2),
            nn.ReLU(),
            nn.Linear(embedding_size // 2, hidden_size)
        )
    
    def encode(self, x_seq, mask):

        out = self.encoder(x_seq)   # [B, T_img, 2H]

        out = out * mask.unsqueeze(-1)

        return out

    #     self.encoder = nn.GRU(
    #         input_size=embedding_size,
    #         hidden_size=hidden_size,
    #         num_layers=1,
    #         batch_first=True,
    #         bidirectional=True
    #     )

    # def encode(self, x_seq, mask):

    #     lengths = mask.sum(dim=1).long().cpu()

    #     packed_x = pack_padded_sequence(
    #         x_seq,
    #         lengths,
    #         batch_first=True,
    #         enforce_sorted=False
    #     )

    #     packed_out, _ = self.encoder(packed_x)

    #     out, _ = pad_packed_sequence(
    #         packed_out,
    #         batch_first=True,
    #         total_length=x_seq.size(1)
    #     ) 

    #     return out

class ImageExpert(nn.Module):

    def __init__(self, embedding_img, hidden_size):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(embedding_img, embedding_img // 2),
            nn.ReLU(),
            nn.Linear(embedding_img // 2, hidden_size)
        )

    def encode(self, img_seq, mask):

        feat = self.encoder(img_seq)

        feat = feat * mask.unsqueeze(-1)

        return feat

