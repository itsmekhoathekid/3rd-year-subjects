import torch
from torch import nn
import numpy as np

class ScaledDotProductAttention(nn.Module):
    def __init__(self, head: int, d_model: int):
        super(ScaledDotProductAttention, self).__init__()

        self.d_model = d_model
        self.d_q = d_model // head
        self.d_kv = d_model // head
        self.head = head

        self.fc_q = nn.Linear(d_model, head * self.d_q)
        self.fc_k = nn.Linear(d_model, head * self.d_kv)
        self.fc_v = nn.Linear(d_model, head * self.d_kv)

    def forward(self, queries, keys, values, attention_mask):
        b_s, nq = queries.shape[:2]
        nk = keys.shape[1]
        q = self.fc_q(queries).view(b_s, nq, self.head, self.d_q).permute(0, 2, 1, 3)   # (b_s, h, nq, d_q)
        k = self.fc_k(keys).view(b_s, nk, self.head, self.d_kv).permute(0, 2, 3, 1)     # (b_s, h, nk, d_kv)
        v = self.fc_v(values).view(b_s, nk, self.head, self.d_kv).permute(0, 2, 1, 3)   # (b_s, h, nk, d_kv)

        att = torch.matmul(q, k) / np.sqrt(self.d_kv)  # (b_s, h, nq, nk)
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)
            att.masked_fill(attention_mask, -1e4)
        att = torch.softmax(att, dim=-1)
        output = torch.matmul(att, v).permute(0, 2, 1, 3).reshape(b_s, -1, self.d_model)

        return output