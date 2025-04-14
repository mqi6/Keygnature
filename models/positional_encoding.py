import math
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    """
    Implements the standard sinusoidal positional encoding.
    """
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            # For odd d_model, handle last dimension separately.
            pe[:, 1::2] = torch.cos(position * div_term[:pe[:, 1::2].shape[1]])
        pe = pe.unsqueeze(0)  # shape [1, max_len, d_model]
        self.register_buffer("pe", pe)
        
    def forward(self, x):
        """
        Input x: [batch, seq_len, d_model]
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

if __name__ == "__main__":
    pe = PositionalEncoding(d_model=128, dropout=0.1, max_len=100)
    x = torch.zeros(32, 50, 128)
    out = pe(x)
    print(out.shape)  # Expected output: [32, 50, 128]
