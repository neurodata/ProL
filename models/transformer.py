'''
Reference: https://nlp.seas.harvard.edu/annotated-transformer/
'''

import torch
import torch.nn as nn
import math

class TemporalEncodings():
    def sinusoid(self, d_model, max_len, C):
        # original sinusoidal encoding
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(C) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        return pe

    def modulo(self, d_model, max_len, C=None):
        # modulo encoding
        position = torch.arange(max_len).unsqueeze(1)
        L = torch.arange(1, d_model + 1, 1) * 5
        me = torch.fmod(position, L).unsqueeze(0) / L
        return me

    def randomized(self, d_model, max_len, C=None):
        # randomized encoding
        position = torch.arange(max_len).unsqueeze(1)
        w = torch.normal(0, 1, (1, d_model))
        b = 2 * math.pi * torch.rand(1, d_model)
        re = torch.cos(position * w + b).unsqueeze(0) * math.sqrt(2/d_model)
        return re

    def fourier(self, d_model, max_len, C=None):
        # frequency-adjusted fourier encoding
        position = torch.arange(max_len).unsqueeze(1)
        div_term = 2 * math.pi / torch.arange(1, d_model//2 + 1, 1)
        ffe = torch.zeros(1, max_len, d_model)
        ffe[0, :, 0::2] = torch.sin(position * div_term)
        ffe[0, :, 1::2] = torch.cos(position * div_term)
        return ffe

class AttentionLayer(nn.Module):
    def __init__(self, d_model=512, num_heads=8, hidden_dim=2048, p=0.1):
        super().__init__()
        self.p = p
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(hidden_dim, d_model),
        )
        self.mha = nn.MultiheadAttention(d_model, num_heads, p)
        self.layernorm1 = nn.LayerNorm(normalized_shape=d_model, eps=1e-6)
        self.layernorm2 = nn.LayerNorm(normalized_shape=d_model, eps=1e-6)
        self.dropout1 = nn.Dropout(p)
        self.dropout2 = nn.Dropout(p)

    def forward(self, x):
        attn_output, _ = self.mha(x, x, x)
        out1 = self.layernorm1(x + self.dropout1(attn_output))
        ff_output = self.feed_forward(out1)
        out2 = self.layernorm2(out1 + self.dropout2(ff_output))
        return out2

class TransformerClassifier(nn.Module):
    def __init__(self, cfg, input_size, num_classes, p=0.1):
        super().__init__()
        
        self.aggregate_type = cfg.net.aggregate_type
        encoder_type = cfg.net.encoder_type
        d_model = cfg.net.d_model
        num_heads = cfg.net.num_heads
        ff_hidden_dim = cfg.net.ff_hidden_dim
        num_attn_blocks = cfg.net.num_attn_blocks
        max_len = cfg.net.max_len
        C = cfg.net.C

        self.cfg = cfg
        self.input_size = input_size

        self.attention_blocks = nn.ModuleList(
            [AttentionLayer(d_model, num_heads, ff_hidden_dim) for _ in range(num_attn_blocks)]
        )
        self.layernorm = nn.LayerNorm(normalized_shape=d_model, eps=1e-6)
        self.classifier = nn.Linear(d_model, num_classes)
        self.dropout = nn.Dropout(p)

        TempEnc = TemporalEncodings()
        encoding = None
        try:
            encoding = getattr(TempEnc, encoder_type)
        except AttributeError:
            raise NotImplementedError
        
        if self.aggregate_type == 'concat':
            self.input_embedding = nn.Linear(input_size+1, d_model // 2)
            te = encoding(d_model//2, max_len, C)
        else:
            self.input_embedding = nn.Linear(input_size+1, d_model)
            te = encoding(d_model, max_len, C)

        self.register_buffer('te', te)

    def temporal_encoder(self, t):
        enc = torch.cat([self.te[:, t[i].squeeze().long(), :] for i in range(t.size(0))])
        return enc
        
    def forward(self, z):
        x, t, y = torch.split(z, [self.input_size, 1, 1], dim=-1)
        u = torch.cat((x, y), dim=-1)
        u = self.input_embedding(u)

        # temporal encoding
        t = self.temporal_encoder(t)

        # type of aggregation
        if self.aggregate_type == 'concat':
            x = torch.cat((u, t), dim=-1)
        else:
            x = self.dropout(u + t)
        
        for attn_block in self.attention_blocks:
            x = attn_block(x)
        x = self.layernorm(x)

        x = torch.select(x, 1, x.shape[1]-1)
        x = self.classifier(x)
        return x