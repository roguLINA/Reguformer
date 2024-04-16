import torch
import torch.nn as nn
import torch.nn.functional as F

import math


def scaled_dot_product(q, k, v, mask=None):
    d_k = q.size()[-1]
    attn_logits = torch.matmul(q, k.transpose(-2, -1))
    attn_logits = attn_logits / math.sqrt(d_k)
    if mask is not None:
        attn_logits = attn_logits.masked_fill(mask == 0, -9e15)
    attention = F.softmax(attn_logits, dim=-1)
    values = torch.matmul(attention, v)
    return values, attention

def scaled_dot_product_Lipschitz_regularization(q, k, v, x, alpha):
    inf_2_norm = lambda a: torch.max(torch.sqrt(torch.sum(torch.pow(a, 2), dim=1)))

    q_norm_2 = torch.pow(torch.norm(q, p=2, dim=-1, keepdim=True), 2)
    qk = torch.matmul(q, k.transpose(-2, -1))
    k_norm_2 = torch.pow(torch.norm(k, p=2, dim=-1, keepdim=True).transpose(-2, -1), 2)


    attn_logits = q_norm_2 - 2 * qk + k_norm_2
    attn_logits = attn_logits / torch.norm(q, p="fro") / inf_2_norm(x.transpose(-2, -1))
    attention = F.softmax(-alpha * attn_logits, dim=-1)
    values = torch.matmul(attention, v)
    return values, attention


class MultiheadAttention(nn.Module):
    def __init__(self, input_dim, embed_dim, n_heads, head_attention="MHSA", alpha=None):
        super().__init__()
        # assert (
        #     embed_dim % n_heads == 0
        # ), "Embedding dimension must be 0 modulo number of heads."

        self.head_attention = head_attention
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        self.alpha = alpha

        # Stack all weight matrices 1...h together for efficiency
        # Note that in many implementations you see "bias=False" which is optional
        self.qkv_proj = nn.Linear(input_dim, 3 * embed_dim)
        self.o_proj = nn.Linear(embed_dim, embed_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        # Original Transformer initialization, see PyTorch documentation
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        self.qkv_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)

    def forward(self, x, mask=None, return_attention=False):
        # print('x.shape', x.shape)
        batch_size, seq_length, _ = x.size()
        # print('batch_size', batch_size)
        # print('seq_length', seq_length)
        qkv = self.qkv_proj(x)

        # Separate Q, K, V from linear output
        qkv = qkv.reshape(batch_size, seq_length, self.n_heads, 3 * self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3)  # [Batch, Head, SeqLen, Dims]
        q, k, v = qkv.chunk(3, dim=-1)

        # Determine value outputs
        if self.head_attention == "MHSA":
            values, attention = scaled_dot_product(q, k, v, mask=mask)
        elif self.head_attention == "LRSA":
            values, attention = scaled_dot_product_Lipschitz_regularization(q, k, v, x, alpha=self.alpha)
        values = values.permute(0, 2, 1, 3)  # [Batch, SeqLen, Head, Dims]
        values = values.reshape(batch_size, seq_length, self.embed_dim)
        o = self.o_proj(values)

        if return_attention:
            return o, attention
        else:
            return o


class EncoderBlock(nn.Module):
    def __init__(self, input_dim, n_heads, d_ff, head_attention, alpha, dropout=0.0, activation="relu"):
        """
        Inputs:
            input_dim - Dimensionality of the input
            n_heads - Number of heads to use in the attention block
            d_ff - Dimensionality of the hidden layer in the MLP
            dropout - Dropout probability to use in the dropout layers
        """
        super().__init__()

        # Attention layer
        self.self_attn = MultiheadAttention(
            input_dim=input_dim, 
            embed_dim=input_dim, 
            n_heads=n_heads, 
            head_attention=head_attention, 
            alpha=alpha,
        ) 
        self.activation = nn.ReLU(inplace=True) if activation == "relu" else nn.GELU()

        # Two-layer MLP
        self.linear_net = nn.Sequential(
            nn.Linear(input_dim, d_ff),
            nn.Dropout(dropout),
            self.activation,
            nn.Linear(d_ff, input_dim),
        )

        # Layers to apply in between the main layers
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Attention part
        attn_out = self.self_attn(x, mask=mask)
        x = x + self.dropout(attn_out)
        x = self.norm1(x)

        # MLP part
        linear_out = self.linear_net(x)
        x = x + self.dropout(linear_out)
        x = self.norm2(x)

        return x


class LRformerEncoder(nn.Module):
    def __init__(self, e_layers, **block_args):
        super().__init__()
        self.layers = nn.ModuleList(
            [EncoderBlock(**block_args) for _ in range(e_layers)]
        )

    def forward(self, x, mask=None):
        for l in self.layers:
            x = l(x, mask=mask)
        return x

    def get_attention_maps(self, x, mask=None):
        attention_maps = []
        for l in self.layers:
            _, attn_map = l.self_attn(x, mask=mask, return_attention=True)
            attention_maps.append(attn_map)
            x = l(x)
        return attention_maps


# class PositionalEncoding(nn.Module):
#     def __init__(self, d_model, max_len=5000):
#         """
#         Inputs
#             d_model - Hidden dimensionality of the input.
#             max_len - Maximum length of a sequence to expect.
#         """
#         super().__init__()

#         # Create matrix of [SeqLen, HiddenDim] representing the positional encoding for max_len inputs
#         pe = torch.zeros(max_len, d_model)
#         position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
#         div_term = torch.exp(
#             torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
#         )
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#         pe = pe.unsqueeze(0)

#         # register_buffer => Tensor which is not a parameter, but should be part of the modules state.
#         # Used for tensors that need to be on the same device as the module.
#         # persistent=False tells PyTorch to not add the buffer to the state dict (e.g. when we save the model)
#         self.register_buffer("pe", pe, persistent=False)

#     def forward(self, x):
#         x = x + self.pe[:, : x.size(1)]
#         return x
