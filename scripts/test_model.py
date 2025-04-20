# test_model.py

import torch
from torch.utils.data import DataLoader
from preprocess.dataset import SessionDataset
from models.dual_transformer import DualStreamModel

def test_forward_pass(data_dir="data/raw", batch_size=2):
    # Load one small batch from the dataset.
    dataset = SessionDataset(data_dir=data_dir, segment_duration=10, mouse_sample_rate=100,
                             max_mouse_len=1000, max_key_len=500, debug=False)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    # Get a single batch.
    sample_batch = next(iter(dataloader))
    mouse_batch = sample_batch["mouse"]  # shape [batch, 1000, 5]
    key_batch = sample_batch["keyboard"] # shape [batch, 500, 4]
    
    print("Loaded batch shapes:")
    print(" Mouse:", mouse_batch.shape)
    print(" Keyboard:", key_batch.shape)
    print(" Player IDs:", sample_batch["player_id"])
    
    # Initialize the model.
    mouse_input_dim = 5
    keyboard_input_dim = 4
    embed_dim = 128
    n_layers = 2     # Use a smaller model for debugging
    n_heads = 4
    dropout = 0.1
    max_len = 1000  # maximum length used for positional encoding
    
    model = DualStreamModel(mouse_input_dim, keyboard_input_dim,
                            embed_dim, n_layers, n_heads, dropout, max_len)
    
    # Put the model in evaluation mode.
    model.eval()
    
    # Run the forward pass.
    with torch.no_grad():
        embedding = model(mouse_batch, key_batch)
    
    print("Output embedding shape:", embedding.shape)
    print("Output embedding (first sample):", embedding[0])
    # Check the norm of the output (should be ~1 due to normalization).
    norms = torch.norm(embedding, p=2, dim=1)
    print("Embedding norms:", norms)

if __name__ == "__main__":
    test_forward_pass()
