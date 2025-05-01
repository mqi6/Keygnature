import torch
from torch import nn

class PositionalEncoding(nn.Module):
    def __init__(self, k, d_model, seq_len):
        super().__init__()
        self.embedding = nn.Parameter(torch.zeros([k, d_model]), requires_grad=True)
        nn.init.xavier_uniform_(self.embedding, gain=1)
        positions = torch.arange(seq_len).unsqueeze(1).repeat(1, k)
        self.register_buffer('positions', positions)
        interval = seq_len / k
        mu_list = [nn.Parameter(torch.tensor(i * interval)) for i in range(k)]
        self.mu = nn.Parameter(torch.stack(mu_list).unsqueeze(0))  # [1 x k]
        self.sigma = nn.Parameter(torch.full((1, k), 50.0))       # [1 x k]

    def normal_pdf(self, pos, mu, sigma):
        a = pos - mu
        logp = -a**2 / (2 * sigma**2) - torch.log(sigma)
        return torch.softmax(logp, dim=1)

    def forward(self, x):
        # x: [batch x seq_len x d_model]
        pdfs = self.normal_pdf(self.positions, self.mu, self.sigma)  # [seq_len x k]
        pe = pdfs @ self.embedding                                   # [seq_len x d_model]
        pe = pe.unsqueeze(0).expand_as(x)
        return x + pe

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, heads, _heads, dropout, seq_len):
        super().__init__()
        self.attn1 = nn.MultiheadAttention(d_model, heads, batch_first=True)
        self.attn2 = nn.MultiheadAttention(seq_len, _heads, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.cnn = nn.Sequential(
            nn.Conv2d(1, seq_len, (1,1)), nn.BatchNorm2d(seq_len), nn.Dropout(dropout), nn.ReLU(),
            nn.Conv2d(seq_len, seq_len, (3,3), padding=1), nn.BatchNorm2d(seq_len), nn.Dropout(dropout), nn.ReLU(),
            nn.Conv2d(seq_len, 1, (5,5), padding=2), nn.BatchNorm2d(1), nn.Dropout(dropout), nn.ReLU()
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, src):
        a1, _ = self.attn1(src, src, src)
        a2, _ = self.attn2(src.transpose(-1,-2), src.transpose(-1,-2), src.transpose(-1,-2))
        a2 = a2.transpose(-1, -2)
        x = self.norm1(src + a1 + a2)
        c = self.cnn(x.unsqueeze(1)).squeeze(1)
        return self.norm2(x + c)

class TransformerEncoder(nn.Module):
    def __init__(self, d_model, heads, _heads, seq_len, num_layers=2, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, heads, _heads, dropout, seq_len)
            for _ in range(num_layers)
        ])

    def forward(self, src):
        for layer in self.layers:
            src = layer(src)
        return src

class KeystrokeTransformer(nn.Module):
    def __init__(self, num_layers, d_model, k, heads, _heads, seq_len, trg_len, dropout):
        super().__init__()
        self.pos = PositionalEncoding(k, d_model, seq_len)
        self.enc = TransformerEncoder(d_model, heads, _heads, seq_len, num_layers, dropout)
        self.ff = nn.Sequential(
            nn.Linear(d_model * seq_len, d_model * seq_len // 2),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(d_model * seq_len // 2, trg_len),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.pos(x)
        x = self.enc(x)
        out = self.ff(x.flatten(1))
        # L2 normalize embeddings with epsilon for stability
        return nn.functional.normalize(out, p=2, dim=1, eps=1e-6)
