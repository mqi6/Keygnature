# test_triplet.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from collections import Counter

from preprocess.dataset import SessionDataset
from models.dual_transformer import DualStreamModel

def main():
    # 1) Load the entire dataset
    dataset = SessionDataset(
        data_dir="data/raw",
        segment_duration=10,
        mouse_sample_rate=100,
        max_mouse_len=1000,
        max_key_len=500,
        debug=False
    )
    print(f"Found {len(dataset)} sessions in dataset.")
    
    loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False, num_workers=0)
    batch = next(iter(loader))
    player_ids = batch["player_id"]
    
    # 2) Count sessions per player
    counts = Counter(player_ids)
    print("Session counts by player:", counts)
    
    # 3) Pick anchor-player (first with ≥2 sessions) and a different negative-player
    anchor_player = None
    for p, c in counts.items():
        if c >= 2:
            anchor_player = p
            break
    if anchor_player is None:
        raise RuntimeError("No player has at least 2 sessions—cannot form a positive pair.")
    negative_player = next(p for p in counts if p != anchor_player)
    
    print(f"Anchor player: {anchor_player}  (has {counts[anchor_player]} sessions)")
    print(f"Negative player: {negative_player}  (has {counts[negative_player]} sessions)")
    
    # 4) Find indices in the batch
    idx_anchor   = player_ids.index(anchor_player)
    idx_positive = player_ids.index(anchor_player, idx_anchor + 1)
    idx_negative = player_ids.index(negative_player)
    print(f"indices → anchor: {idx_anchor}, positive: {idx_positive}, negative: {idx_negative}")
    
    # 5) Extract those three samples
    mouse = batch["mouse"]       # shape [N,1000,5]
    key   = batch["keyboard"]    # shape [N,500,4]
    mouse_a = mouse[idx_anchor].unsqueeze(0)
    key_a   = key[idx_anchor].unsqueeze(0)
    mouse_p = mouse[idx_positive].unsqueeze(0)
    key_p   = key[idx_positive].unsqueeze(0)
    mouse_n = mouse[idx_negative].unsqueeze(0)
    key_n   = key[idx_negative].unsqueeze(0)
    
    # 6) Instantiate a small model
    model = DualStreamModel(
        mouse_input_dim=5, keyboard_input_dim=4,
        embed_dim=128, n_layers=2, n_heads=4,
        dropout=0.1, max_len=1000
    )
    model.eval()
    
    # 7) Compute embeddings
    with torch.no_grad():
        emb_a = model(mouse_a, key_a)  # [1,128]
        emb_p = model(mouse_p, key_p)
        emb_n = model(mouse_n, key_n)
    
    # 8) Compute distances and loss
    dist = nn.PairwiseDistance(p=2)
    d_ap = dist(emb_a, emb_p).item()
    d_an = dist(emb_a, emb_n).item()
    margin = 1.0
    loss_fn = nn.TripletMarginLoss(margin=margin, p=2)
    loss = loss_fn(emb_a, emb_p, emb_n).item()
    
    print(f"Distance(A,P): {d_ap:.4f}   Distance(A,N): {d_an:.4f}   Margin: {margin}")
    print(f"Triplet loss: {loss:.4f}")
    
    # 9) If loss==0, test smaller margins
    if loss == 0.0:
        for m in [0.5, 0.2, 0.1]:
            tmp = nn.TripletMarginLoss(margin=m, p=2)(emb_a, emb_p, emb_n).item()
            print(f" Margin={m:>3}: loss={tmp:.4f}")
        print("→ All zero. You need harder negatives (e.g. batch-hard mining) or a smaller margin.")
    else:
        print("✔ Non-zero loss achieved — triplet pipeline is working!")

if __name__ == "__main__":
    main()
