import os
import torch
from torch.utils.data import Dataset
from .parse_logs import parse_log
from .feature_engineering import compute_mouse_velocity, compute_keyboard_hold_times
import numpy as np

def extract_player_and_session(filename):
    """
    Extract player_id and session_id from filename.
    Expected filename format: playerX_sessionY.csv
    """
    basename = os.path.basename(filename)
    name, _ = os.path.splitext(basename)
    parts = name.split('_')
    player_id = parts[0]  # e.g., "player1"
    session_id = parts[1] if len(parts) > 1 else "session0"
    return player_id, session_id

def pad_sequence(seq, max_len, feature_dim):
    """
    Pad a sequence (list of feature vectors) with zeros to a fixed length.
    If sequence is longer than max_len, it will be cropped.
    Returns a numpy array of shape (max_len, feature_dim).
    """
    seq = np.array(seq)
    seq_len = seq.shape[0]
    if seq_len >= max_len:
        return seq[:max_len]
    else:
        pad = np.zeros((max_len - seq_len, feature_dim))
        return np.vstack([seq, pad])

class SessionDataset(Dataset):
    def __init__(self, data_dir, segment_duration=10, mouse_sample_rate=100, max_mouse_len=1000, max_key_len=500):
        """
        data_dir: directory containing session CSV files.
        segment_duration: seconds per segment (currently, each file is one segment).
        mouse_sample_rate: sample rate for mouse events.
        max_mouse_len: maximum length for mouse events.
        max_key_len: maximum length for keyboard events.
        """
        self.data_dir = data_dir
        self.files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.csv')]
        self.segment_duration = segment_duration
        self.mouse_sample_rate = mouse_sample_rate
        self.max_mouse_len = max_mouse_len
        self.max_key_len = max_key_len

        # Prepare samples as a list of tuples (file_path, player_id, session_id)
        self.samples = []
        for f in self.files:
            player_id, session_id = extract_player_and_session(f)
            self.samples.append((f, player_id, session_id))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        file_path, player_id, session_id = self.samples[idx]
        data = parse_log(file_path)
        # Process events
        mouse_events = compute_mouse_velocity(data["mouse"])
        keyboard_events = compute_keyboard_hold_times(data["keyboard"])
        # Convert to feature vectors.
        # Mouse: [timestamp, x, y, vx, vy]
        mouse_feats = []
        for e in mouse_events:
            if e["x"] is not None and e["y"] is not None:
                mouse_feats.append([e["timestamp"], e["x"], e["y"], e["vx"], e["vy"]])
        # Keyboard: [timestamp, key (numeric), action (press=1, release=0), hold_time]
        key_feats = []
        for e in keyboard_events:
            try:
                key_val = float(ord(e["key"][0])) if e["key"] else 0.0
            except Exception:
                key_val = 0.0
            action_val = 1.0 if e["action"] == "press" else 0.0
            hold_time = e.get("hold_time", 0.0)
            key_feats.append([e["timestamp"], key_val, action_val, hold_time])
        # Normalize timestamps relative to session start:
        if len(mouse_feats) > 0:
            start_time = mouse_feats[0][0]
            mouse_feats = [[x - start_time if i == 0 else x for i, x in enumerate(feat)] for feat in mouse_feats]
        if len(key_feats) > 0:
            start_time = key_feats[0][0]
            key_feats = [[x - start_time if i == 0 else x for i, x in enumerate(feat)] for feat in key_feats]
        
        # Pad sequences:
        mouse_arr = pad_sequence(mouse_feats, self.max_mouse_len, feature_dim=5)
        key_arr = pad_sequence(key_feats, self.max_key_len, feature_dim=4)

        # Convert to torch tensors:
        mouse_tensor = torch.tensor(mouse_arr, dtype=torch.float32)
        key_tensor = torch.tensor(key_arr, dtype=torch.float32)

        return {"mouse": mouse_tensor, "keyboard": key_tensor, "player_id": player_id, "session_id": session_id}

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/raw")
    args = parser.parse_args()
    
    dataset = SessionDataset(args.data_dir)
    print("Total sessions:", len(dataset))
    sample = dataset[0]
    print("Sample keys:", sample.keys())
    print("Mouse tensor shape:", sample["mouse"].shape)
    print("Keyboard tensor shape:", sample["keyboard"].shape)
