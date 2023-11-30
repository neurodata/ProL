import torch
import torch.nn as nn
import math

class AttentionLayer(nn.Module):
    def __init__(self, d_model, num_heads, hidden_dim, p=0.1):
        super().__init__()
        
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, hidden_dim, p),
            nn.ReLU(),
            nn.Linear(hidden_dim, d_model, p),
        )
        self.mha = nn.MultiheadAttention(d_model, num_heads, p)
        self.layernorm1 = nn.LayerNorm(normalized_shape=d_model, eps=1e-6)
        self.layernorm2 = nn.LayerNorm(normalized_shape=d_model, eps=1e-6)

    def forward(self, x):
        attn_output, _ = self.mha(x, x, x)
        out1 = self.layernorm1(x + attn_output)
        ff_output = self.feed_forward(out1)
        out2 = self.layernorm2(out1 + ff_output)
        return out2

class TransformerClassifier(nn.Module):
    def __init__(self, cfg, input_size, d_model, num_heads, ff_hidden_dim, num_attn_blocks=1, num_classes=2, 
                 contextlength=200, C=100.0, max_len=5000):
        super().__init__()
        self.cfg = cfg
        self.input_size = input_size
        self.d_model = d_model
        self.attention_blocks = nn.ModuleList(
            [AttentionLayer(d_model, num_heads, ff_hidden_dim) for _ in range(num_attn_blocks)]
        )
        self.input_embedding = nn.Linear(input_size+1, d_model // 2)
        self.layernorm = nn.LayerNorm(normalized_shape=d_model, eps=1e-6)
        self.classifier = nn.Linear(d_model, num_classes)

        # sinusoidal encoding
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model//2, 2) * (-math.log(C) / (d_model//2)))
        pe = torch.zeros(1, max_len, d_model//2)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

        # modulo encoding
        position = torch.arange(max_len).unsqueeze(1)
        L = torch.arange(1, d_model//2+1, 1) * 5
        me = torch.fmod(position, L).unsqueeze(0) / L
        self.register_buffer('me', me)

    def sinusoid_encoder(self, t):
        pos_enc = torch.cat([self.pe[:, t[i].squeeze().long(), :] for i in range(t.size(0))])
        return pos_enc

    def modulo_encoder(self, t):
        mod_enc = torch.cat([self.me[:, t[i].squeeze().long(), :] for i in range(t.size(0))])
        return mod_enc
        
    def forward(self, z):
        x, t, y = torch.split(z, [self.input_size, 1, 1], dim=-1)
        u = torch.cat((x, y), dim=-1)
        u = self.input_embedding(u)

        # type of temporal encoder
        if self.cfg.encoder_type == 'sinusoid':
            t = self.sinusoid_encoder(t)
        elif self.cfg.encoder_type == 'modulo':
            t = self.modulo_encoder(t)
        else:
            t = self.modulo_encoder(t)

        # type of aggregation
        # x = self.layernorm(u + t)
        x = torch.cat((u, t), dim=-1)

        for attn_block in self.attention_blocks:
            x = attn_block(x)
        x = torch.select(x, 1, x.shape[1]-1)
        x = self.classifier(x)
        return x