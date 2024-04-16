import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

import math

from copy import deepcopy
from scipy.stats import bernoulli


class TriangularCausalMask():
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask
    

class ProbMask():
    def __init__(self, B, H, L, index, scores, device="cpu"):
        _mask = torch.ones(L, scores.shape[-1], dtype=torch.bool).to(device).triu(1)
        _mask_ex = _mask[None, None, :].expand(B, H, L, scores.shape[-1])
        indicator = _mask_ex[torch.arange(B)[:, None, None],
                             torch.arange(H)[None, :, None],
                             index, :].to(device)
        self._mask = indicator.view(scores.shape).to(device)
    
    @property
    def mask(self):
        return self._mask
    

class ConvLayer(nn.Module):
    def __init__(self, c_in):
        super(ConvLayer, self).__init__()
        padding = 1 if torch.__version__ >= "1.5.0" else 2
        self.downConv = nn.Conv1d(
            in_channels=c_in,
            out_channels=c_in,
            kernel_size=3,
            padding=padding,
            padding_mode="circular",
        )
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()
        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.downConv(x.permute(0, 2, 1))
        x = self.norm(x)
        x = self.activation(x)
        x = self.maxPool(x)
        x = x.transpose(1, 2)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None):
        # x [B, L, D]
        new_x, attn = self.attention(x, x, x, attn_mask=attn_mask)
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y), attn
    
# class DropDim(nn.Module):
#     def __init__(self, p, mode="train"):
#         super(DropDim, self).__init__()
#         self.p = p
#         self.mode = mode

#     def forward(self, h):
#         # print(self.mode)
#         if self.mode == "inference":
#             return h
#         B, T, D = h.shape
#         print('!', end='')
#         new_h = torch.tensor([], device=h.device)
#         for i in range(B):
#             el = deepcopy(h[i, :, :].detach())
#             ksi = bernoulli.rvs(p=0.3, size=D)
#             el[:, ksi == 0] = 0
#             new_h = torch.cat([new_h, el[None, :, :]])
#         return new_h
    

class DropDim(nn.Module):
    def __init__(self, p, type, mode="train", alpha=0):
        super(DropDim, self).__init__()
        self.p = p
        self.mode = mode
        self.type = type
        if self.type == "random":
            self.alpha = None
        else:
            self.alpha = alpha

    def forward(self, h):
        # print(self.mode)
        if self.mode == "inference":
            return h
        
        B, T, D = h.shape
        # print('D', D)
        new_h = torch.tensor([], device=h.device)
        for i in range(B):
            el = deepcopy(h[i, :, :].detach())

            if self.type == "random":
                ksi = bernoulli.rvs(p=0.3, size=D)
                el[:, ksi == 0] = 0
                new_h = torch.cat([new_h, el[None, :, :]])
            elif self.type == "span":
                l = torch.round(self.alpha * torch.rand(1))[0].long().item() # (r1 - r2) * torch.rand(a, b) + r2, a,b -- dimensions, r1, r2 -- range for distribution
                s = np.random.choice(np.arange(D - l))
                el[:, s:s+l-1] = 0
                new_h = torch.cat([new_h, el[None, :, :]])
        return new_h
        

class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None, p=None, alpha=0, drop_dim_type="random"):
        super(Encoder, self).__init__()
        self.p = p
        self.alpha = alpha
        self.drop_dim_type = drop_dim_type
        if p is not None: 
            self.drop_dim = DropDim(p=self.p, type=self.drop_dim_type, alpha=self.alpha)
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = (
            nn.ModuleList(conv_layers) if conv_layers is not None else None
        )
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        # x [B, L, D]
        attns = []
        if self.conv_layers is not None:
            for attn_layer, conv_layer in zip(self.attn_layers, self.conv_layers):
                x, attn = attn_layer(x, attn_mask=attn_mask)
                x = conv_layer(x)
                if self.p is not None: 
                    x = self.drop_dim(x)

                attns.append(attn)
            x, attn = self.attn_layers[-1](x, attn_mask=attn_mask)
            if self.p is not None: 
                x = self.drop_dim(x)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask)
                attns.append(attn)
                if self.p is not None: 
                    x = self.drop_dim(x)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns


class FullAttention(nn.Module):
    def __init__(
        self,
        mask_flag=False,
        factor=5,
        scale=None,
        attention_dropout=0.1,
        output_attention=False,
        sparsification_type=None, 
    ):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1.0 / math.sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)


class ProbAttention(nn.Module):
    def __init__(
        self,
        mask_flag=True,
        factor=5,
        scale=None,
        attention_dropout=0.1,
        output_attention=False,
        sparsification_type='topQ',
    ):
        """ProbSparseAttention implementation.

        :param mask_flag: if True implement mask for attention   
        :param factor: hyperparameter which influence the number of required queries/keys: factor * ln(seq_len)
        :param scale: if is not None is used for the resulting attention matrix scaling   
        :param attention_dropout: dropout for attention   
        :param output_attention: if True returns attention matrix
        :param sparsification_type: regularization strategy: 
                    - topQ -- consider top queries
                    - topK -- consider top keys
                    - randQ -- consider random queries
                    - randK -- consider random keys
                    - topQ_topK -- consider top queries and top keys
                    - topQ_randK -- consider top queries and random keys
                    - randQ_topK -- consider random queries and top keys
                    - randQ_randK -- consider random queries and random keys
        
        """
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        self.sparsification_type = sparsification_type

    def _prob_QK(self, Q, K, sample_k, n_top):  # n_top: c*ln(L_q)
        B, H, L_K, E = K.shape # B=64, H=4 L=100 E=16
        _, _, L_Q, _ = Q.shape

        if 'Q' in self.sparsification_type:
            # calculate the sampled Q_K
            if 'topQ' in self.sparsification_type:
                K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E) # [B, H, L, L, E]
                index_sample = torch.randint(
                    L_K, (L_Q, sample_k)
                ) # [L, sample_K]
                index_sample, _ = index_sample.sort()
                K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :] # [B, H, L, sample_K, E]
                Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze(-2) # [B, H, L, sample_K]

                # find the Top_k query with sparisty measurement            
                M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
                M_top = M.topk(n_top, sorted=True)[1]
            elif 'randQ' in self.sparsification_type:
                M_top = torch.randint(low=0, high=L_Q-1, size=(B, H, n_top))
                M_top, _ = M_top.sort()

            # use the reduced Q to calculate Q_K
            Q_reduce = Q[
                torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], M_top, :
            ]  # factor*ln(L_q)  [B, H, n_top, E]
        else:
            Q_reduce = Q # [B, H, L, E]

        if 'K' in self.sparsification_type:
            # calculate the sampled Q_K
            if 'topK' in self.sparsification_type:
                if 'Q' not in self.sparsification_type:
                    Q_expand = Q_reduce.unsqueeze(-3).expand(B, H, L_K, L_Q, E) # [B, H, L, L, E]
                    index_sample = torch.randint(
                        L_Q, (L_K, sample_k)
                    ) # [sample_K, L]
                    index_sample, _ = index_sample.sort()
                    Q_sample = Q_expand[:, :, torch.arange(L_K).unsqueeze(1), index_sample, :] # [B, H, L, sample_K, E]
                else:
                    Q_sample = Q_reduce.unsqueeze(-3).expand(B, H, L_K, n_top, E) 
                    
                Q_K_sample = torch.matmul(Q_sample, K.unsqueeze(-1)).squeeze(-1) # [B, H, L, sample_K]
            
                # find the Top_k keys with sparisty measurement
                M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_Q)
                # M_top = M.topk(n_top, sorted=False)[1] # [B, H, n_top]
                M_top = M.topk(n_top, sorted=True)[1]
            elif 'randK' in self.sparsification_type:
                M_top = torch.randint(low=0, high=L_Q-1, size=(B, H, n_top))
                M_top, _ = M_top.sort()

            # use the reduced Q to calculate Q_K
            K_reduce = K[
                torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], M_top, :
            ]  
        else:
            K_reduce = K

        Q_K = torch.matmul(Q_reduce, K_reduce.transpose(-2, -1)) #[B, H, L, n_top]
        self.M_top = M_top

        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape
        if not self.mask_flag:
            V_sum = V.mean(dim=-2)
            contex = V_sum.unsqueeze(-2).expand(B, H, L_Q, V_sum.shape[-1]).clone()
        else:  # use mask
            assert L_Q == L_V  # requires that L_Q == L_V, i.e. for self-attention only
            contex = V.cumsum(dim=-2)
        return contex

    def _update_context(self, context_in, V, scores, index, L_Q, attn_mask):
        B, H, L_V, D = V.shape

        if self.mask_flag:
            attn_mask = ProbMask(B, H, L_Q, index, scores, device=V.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        attn = torch.softmax(scores, dim=-1) 
        
        if 'Q' in self.sparsification_type:
            context_in[
                torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :
            ] = torch.matmul(attn, V).type_as(context_in)
        else:
            context_in = torch.matmul(attn, V)
            
        if self.output_attention:
            if 'K' in self.sparsification_type:
                return (context_in, attn)
                # raise NotImplementedError("this is not implemented for sparse K")
            attns = (torch.ones([B, H, L_V, L_V]) / L_V).type_as(attn).to(attn.device)
            attns[
                torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :
            ] = attn
            return (context_in, attns)
        else:
            return (context_in, None)

    def forward(self, queries, keys, values, attn_mask):
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape

        queries = queries.transpose(2, 1)
        keys = keys.transpose(2, 1)
        values = values.transpose(2, 1)

        U_part = self.factor * np.ceil(np.log(L_K)).astype("int").item()  # c*ln(L_k)
        u = self.factor * np.ceil(np.log(L_Q)).astype("int").item()  # c*ln(L_q)

        U_part = U_part if U_part < L_K else L_K
        u = u if u < L_Q else L_Q

        scores_top, index = self._prob_QK(queries, keys, sample_k=U_part, n_top=u)
        
        # add scale factor
        scale = self.scale or 1.0 / math.sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale
        # get the context
        context = self._get_initial_context(values, L_Q)
        if 'K' in self.sparsification_type:
            values = values[
                torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :
            ]

        # update the context with selected top_k queries
        context, attn = self._update_context(
            context, values, scores_top, index, L_Q, attn_mask
        )

        return context.transpose(2, 1).contiguous(), attn


class AttentionLayer(nn.Module):
    def __init__(
        self, attention, d_model, n_heads, d_keys=None, d_values=None, mix=False
    ):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads
        self.mix = mix

    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(queries, keys, values, attn_mask)
        if self.mix:
            out = out.transpose(2, 1).contiguous()
        out = out.view(B, L, -1)

        return self.out_projection(out), attn


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        ).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return self.pe[:, : x.size(1)]


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= "1.5.0" else 2
        self.tokenConv = nn.Conv1d(
            in_channels=c_in,
            out_channels=d_model,
            kernel_size=3,
            padding=padding,
            padding_mode="circular",
        )
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_in", nonlinearity="leaky_relu"
                )

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, dropout=0.1):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x)


class DropDimEncoder(nn.Module):
    """Class implements encoding part of DropDim.

    :param enc_in: Size of input embedding
    :param factor: Probsparse attn factor (defaults to 5)
    :param d_model: Dimension of model (defaults to 512)
    :param n_heads: Num of heads (defaults to 8)
    :param e_layers: Num of encoder layers (defaults to 2)
    :param d_ff: Dimension of fcn (defaults to 2048)
    :param dropout: The probability of dropout (defaults to 0.05)
    :param attn: Attention used in encoder (defaults to prob). This can be set to prob (reguformer), full (transformer)
    :param activation: Activation function (defaults to gelu)
    :param output_attention: Whether to output attention in encoder, using this argument means outputing attention (defaults to False)
    :param distil: Whether to use distilling in encoder, using this argument means not using distilling (defaults to True)
    :param device: Device ids of multile gpus (defaults to 0,1,2,3)
    """
    def __init__(
        self,
        enc_in,
        p=None, alpha=0, drop_dim_type="random",
        factor=5,
        d_model=512,
        n_heads=8,
        e_layers=2,
        d_ff=512,
        dropout=0.0,
        attn="prob",
        activation="gelu",
        output_attention=False,
        distil=True,
        device=torch.device("cuda:0"),
        sparsification_type=None,
    ):
        super(DropDimEncoder, self).__init__()

        self.attn = attn
        self.output_attention = output_attention

        # Encoding
        self.enc_embedding = DataEmbedding(enc_in, d_model, dropout)
        # Attention
        Attn = ProbAttention if attn == "prob" else FullAttention

        # Encoder
        self.encoder = Encoder(
            attn_layers=[
                EncoderLayer(
                    AttentionLayer(
                        Attn(
                            False,
                            factor,
                            attention_dropout=dropout,
                            output_attention=output_attention,
                            sparsification_type=sparsification_type,
                        ),
                        d_model,
                        n_heads,
                        mix=False,
                    ),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(e_layers)
            ],
            conv_layers=[ConvLayer(d_model) for l in range(e_layers - 1)] if distil else None,
            norm_layer=torch.nn.LayerNorm(d_model),
            p=p,
        )

    def forward(self, x_enc, enc_self_mask=None):
        """Get sequence embedding.

        :param x_enc: initial representation of sequence.
        :param enc_self_mask: attention mask (we don't use it)

        :return: encoded sequence
        """
        enc_out = self.enc_embedding(x_enc)
        enc_out, _ = self.encoder(enc_out, attn_mask=enc_self_mask)

        return enc_out
    
    def get_attention_maps(self, x, mask=None):
        """
        Get attention maps. Available if self.output_attention=True

        :param x: initial sequence.
        :param mask: attention mask (we don't use it)

        :return: attention maps
        """
        enc_out = self.enc_embedding(x)
        enc_out, enc_attns = self.encoder(enc_out, attn_mask=mask) 

        return enc_attns