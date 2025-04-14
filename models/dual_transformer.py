import torch
import torch.nn as nn
from .positional_encoding import PositionalEncoding

class BaseEncoder(nn.Module):
    def __init__(self, input_dim, embed_dim, n_layers, n_heads, dropout=0.1, max_len=1000):
        super(BaseEncoder, self).__init__()
        self.input_proj = nn.Linear(input_dim, embed_dim)
        self.pos_encoding = PositionalEncoding(embed_dim, dropout, max_len=max_len)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=n_heads,
                                                   dropout=dropout, dim_feedforward=embed_dim*4)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
    
    def forward(self, seq):
        # seq shape: [batch, seq_len, input_dim]
        x = self.input_proj(seq)               # [batch, seq_len, embed_dim]
        x = self.pos_encoding(x)               # add positional encoding
        batch_size = x.size(0)
        # Prepend a [CLS] token
        cls_tok = self.cls_token.expand(batch_size, -1, -1)  # [batch, 1, embed_dim]
        x = torch.cat([cls_tok, x], dim=1)      # [batch, 1+seq_len, embed_dim]
        x = x.transpose(0, 1)                  # Transformer expects [seq_len, batch, embed_dim]
        x = self.encoder(x)                    
        cls_out = x[0]                         # Use output corresponding to the CLS token
        return cls_out

class MouseEncoder(BaseEncoder):
    # Inherits behavior from BaseEncoder (mouse features: 5 dims)
    pass

class KeyboardEncoder(BaseEncoder):
    # Inherits behavior from BaseEncoder (keyboard features: 4 dims)
    pass

class DualStreamModel(nn.Module):
    def __init__(self, mouse_input_dim, keyboard_input_dim, embed_dim=128, n_layers=4, n_heads=8, dropout=0.1, max_len=1000):
        super(DualStreamModel, self).__init__()
        self.mouse_encoder = MouseEncoder(mouse_input_dim, embed_dim, n_layers, n_heads, dropout, max_len)
        self.keyboard_encoder = KeyboardEncoder(keyboard_input_dim, embed_dim, n_layers, n_heads, dropout, max_len)
        self.fusion = nn.Sequential(
            nn.Linear(2 * embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
    
    def forward(self, mouse_seq, keyboard_seq):
        # Encode each modality
        z_mouse = self.mouse_encoder(mouse_seq)      # [batch, embed_dim]
        z_key = self.keyboard_encoder(keyboard_seq)    # [batch, embed_dim]
        z = torch.cat([z_mouse, z_key], dim=1)         # [batch, 2 * embed_dim]
        z = self.fusion(z)                            # [batch, embed_dim]
        # Normalize the final embedding
        z = nn.functional.normalize(z, p=2, dim=1)
        return z

if __name__ == "__main__":
    batch_size = 8
    mouse_seq = torch.randn(batch_size, 500, 5)      # e.g. 500 mouse events, 5 features each
    keyboard_seq = torch.randn(batch_size, 300, 4)     # e.g. 300 keyboard events, 4 features each
    model = DualStreamModel(mouse_input_dim=5, keyboard_input_dim=4,
                            embed_dim=128, n_layers=2, n_heads=4, dropout=0.1, max_len=600)
    out = model(mouse_seq, keyboard_seq)
    print(out.shape)  # Expected: [8, 128]
