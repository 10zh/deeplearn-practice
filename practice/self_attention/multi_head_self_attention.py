import math
import torch
import torch.nn as nn


class MultiHeadSelfAttention(nn.Module):

    def __init__(self, hidden_dim, head_num, attention_dropout=0.1):
        super(MultiHeadSelfAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.head_num = head_num
        self.head_dim = hidden_dim // head_num  # (head_num * head_dim = hidden_dim)

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)  # (hidden_dim,head_dim * head_num)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        self.attention_dropout = nn.Dropout(attention_dropout)

    def forward(self, X, attention_mask=None):
        # X(batch,seq_len,hidden_dim)
        batch, seq_len, _ = X.size()
        Q = self.q_proj(X)
        K = self.k_proj(X)
        V = self.v_proj(X)  # (batch,seq_len,hidden_dim)
        # (batch,seq_len,hidden_dim) => (b,head_num,seq_len,head_dim)
        # (h=>head_num * head_dim)
        q_state = Q.view(batch, seq_len, self.head_num, self.head_dim).transpose(1, 2)
        k_state = K.view(batch, seq_len, self.head_num, self.head_dim).transpose(1, 2)
        v_state = V.view(batch, seq_len, self.head_num, self.head_dim).transpose(1, 2)

        # (b,head_num,s,s)
        attention_weight = torch.matmul(
            q_state, k_state.transpose(-1, 2)  # (b,head_num,s,head_dim)=>(b,head_num,head_dim,s)
        ) / math.sqrt(self.head_dim)
        if attention_mask is not None:
            attention_weight = attention_weight.masked_fill(attention_mask == 0, float('-inf'))
        print(attention_weight.shape)
        attention_weight = torch.softmax(attention_weight, dim=-1)
        attention_weight = self.attention_dropout(attention_weight)
        # (b,head_num,s,head_dim)
        output_mid = torch.matmul(attention_weight, v_state)
        output_mid = output_mid.transpose(1, 2).contiguous()
        output_mid = output_mid.view(batch, seq_len, -1)
        output = self.out_proj(output_mid)
        return output


if __name__ == '__main__':
    attention_mask = (
        torch.tensor(
            [
                [0, 1],
                [0, 0],
                [1, 0]
            ]
        )
        .unsqueeze(1)
        .unsqueeze(2)
        .expand(3, 8, 2, 2)
    )
    print(attention_mask.shape)
    x = torch.rand(3, 2, 128)
    net = MultiHeadSelfAttention(128, 8)  # head_dim = 16
    print(net(x, attention_mask))
