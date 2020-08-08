import torch
from torch import nn

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder,
                 src_pad_idx, trg_pad_idx, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device

    def make_src_mask(self, src):
        # src = [batch size, src len]
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask.to('cpu')  # src_mask = [batch size, 1, 1, src len]

    def make_trg_mask(self, trg):
        # trg = [batch size, trg len]
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(3)
        # trg_pad_mask = [batch size, 1, trg len, 1]
        trg_len = trg.shape[1]
        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device='cpu')).bool()
        # trg_sub_mask = [trg len, trg len]    torch.triu：返回矩阵 下 三角部分，其余部分为0
        trg_mask = trg_pad_mask & trg_sub_mask  # trg_mask = [batch size, 1, trg len, trg len]
        return trg_mask.to('cpu')

    def forward(self, src, trg):
        src = src.to('cpu')
        trg = trg.to('cpu')
        # src = [batch size, src len]    trg = [batch size, trg len]
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        # src_mask = [batch size, 1, 1, src len]
        # trg_mask = [batch size, 1, trg len, trg len]

        enc_src = self.encoder(src, src_mask)  # enc_src = [batch size, src len, hid dim]
        output, attention = self.decoder(trg, enc_src, trg_mask, src_mask)
        # output = [batch size, trg len, output dim]
        # attention = [batch size, n heads, trg len, src len]
        return output, attention
