import argparse
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from preprocess.dataset import SessionDataset, pad_sequence
from models.dual_transformer import DualStreamModel
from utils.config import load_config

def load_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint["config"]
    mouse_input_dim = 5
    keyboard_input_dim = 4
    embed_dim = config["model"]["embed_dim"]
    n_layers = config["model"]["n_layers"]
    n_heads = config["model"]["n_heads"]
    dropout = config["model"].get("dropout", 0.1)
    max_len = max(config["data"].get("max_mouse_len", 1000), config["data"].get("max_key_len", 500))
    model = DualStreamModel(mouse_input_dim, keyboard_input_dim, embed_dim, n_layers, n_heads, dropout, max_len)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, config

def compute_embedding(model, sample, device):
    # sample["mouse"] could be [seq,feat] (test_file) or [1,seq,feat] (DataLoader)
    mouse = sample["mouse"].to(device)
    keyboard = sample["keyboard"].to(device)

    # Ensure batch dimension
    if mouse.dim() == 2:
        mouse = mouse.unsqueeze(0)       # [1, seq_len, feat]
    if keyboard.dim() == 2:
        keyboard = keyboard.unsqueeze(0)

    with torch.no_grad():
        emb = model(mouse, keyboard)     # [1, embed_dim]

    return emb.squeeze(0)               # → [embed_dim]



def cosine_similarity(a, b):
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, config = load_model(args.checkpoint, device)
    
    dataset = SessionDataset(args.data_dir,
                             segment_duration=config["data"].get("segment_duration", 10),
                             mouse_sample_rate=config["data"].get("sample_rate_mouse", 100),
                             max_mouse_len=config["data"].get("max_mouse_len", 1000),
                             max_key_len=config["data"].get("max_key_len", 500))
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    owner_id = args.owner_id
    # Build embeddings per player.
    embeddings_dict = {}
    for sample in dataloader:
        player_id = sample["player_id"][0]
        embedding = compute_embedding(model, sample, device)
        if player_id not in embeddings_dict:
            embeddings_dict[player_id] = []
        embeddings_dict[player_id].append(embedding)
    
    ref_embeddings = {}
    for player, emb_list in embeddings_dict.items():
        stacked = torch.stack(emb_list, dim=0)
        ref_embeddings[player] = torch.mean(stacked, dim=0)
    
    if owner_id not in ref_embeddings:
        print("Owner ID", owner_id, "not found in dataset.")
        return
    owner_ref = ref_embeddings[owner_id]
    
    print("Verification results (cosine similarity to owner's embedding):")
    for player, ref_emb in ref_embeddings.items():
        sim = cosine_similarity(owner_ref, ref_emb)
        print(f"Owner vs {player}: Similarity = {sim:.4f}")
    
    if args.test_file:
        # Process test file.
        from preprocess.parse_logs import parse_log
        from preprocess.feature_engineering import compute_mouse_velocity, compute_keyboard_hold_times
        import numpy as np
        data = parse_log(args.test_file)
        mouse_events = compute_mouse_velocity(data["mouse"])
        keyboard_events = compute_keyboard_hold_times(data["keyboard"])
        mouse_feats = []
        for e in mouse_events:
            if e["x"] is not None and e["y"] is not None:
                mouse_feats.append([e["timestamp"], e["x"], e["y"], e["vx"], e["vy"]])
        key_feats = []
        for e in keyboard_events:
            try:
                key_val = float(ord(e["key"][0])) if e["key"] else 0.0
            except Exception:
                key_val = 0.0
            action_val = 1.0 if e["action"] == "press" else 0.0
            hold_time = e.get("hold_time", 0.0)
            key_feats.append([e["timestamp"], key_val, action_val, hold_time])
        if len(mouse_feats) > 0:
            start_time = mouse_feats[0][0]
            mouse_feats = [[x - start_time if i == 0 else x for i, x in enumerate(feat)] for feat in mouse_feats]
        if len(key_feats) > 0:
            start_time = key_feats[0][0]
            key_feats = [[x - start_time if i == 0 else x for i, x in enumerate(feat)] for feat in key_feats]
        max_mouse_len = config["data"].get("max_mouse_len", 1000)
        max_key_len = config["data"].get("max_key_len", 500)
        mouse_arr = pad_sequence(mouse_feats, max_mouse_len, feature_dim=5)
        key_arr = pad_sequence(key_feats, max_key_len, feature_dim=4)
        mouse_tensor = torch.tensor(mouse_arr, dtype=torch.float32)
        key_tensor = torch.tensor(key_arr, dtype=torch.float32)
        sample = {"mouse": mouse_tensor, "keyboard": key_tensor}
        embedding_test = compute_embedding(model, sample, device)
        best_player = None
        best_sim = -1
        for player, ref_emb in ref_embeddings.items():
            sim = cosine_similarity(embedding_test, ref_emb)
            if sim > best_sim:
                best_sim = sim
                best_player = player
        print(f"Test file '{args.test_file}' is closest to player '{best_player}' with similarity {best_sim:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verification for Dual-Stream Transformer model")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.pth)")
    parser.add_argument("--data_dir", type=str, default="data/raw", help="Directory with raw session CSV files")
    parser.add_argument("--owner_id", type=str, required=True, help="Player ID of the account owner")
    parser.add_argument("--test_file", type=str, default=None, help="Path to a test CSV file for identification")
    args = parser.parse_args()
    main(args)
