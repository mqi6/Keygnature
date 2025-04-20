# preprocess/dataset.py

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
    def __init__(self, data_dir, segment_duration=10, mouse_sample_rate=1000, max_mouse_len=1000, max_key_len=500, debug=False):
        """
        data_dir: directory containing session CSV files.
        segment_duration: seconds per segment (each file is assumed one segment).
        mouse_sample_rate: sample rate for mouse events.
        max_mouse_len: maximum number of mouse events per segment.
        max_key_len: maximum number of keyboard events per segment.
        debug: if True, prints debug statements.
        """
        self.data_dir = data_dir
        self.files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.csv')]
        self.segment_duration = segment_duration
        self.mouse_sample_rate = mouse_sample_rate
        self.max_mouse_len = max_mouse_len
        self.max_key_len = max_key_len
        self.debug = debug

        # Prepare samples as a list of tuples (file_path, player_id, session_id)
        self.samples = []
        for f in self.files:
            player_id, session_id = extract_player_and_session(f)
            self.samples.append((f, player_id, session_id))
        
        if self.debug:
            print(f"[DEBUG] Total CSV files found: {len(self.files)}")
            print(f"[DEBUG] Total samples: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        file_path, player_id, session_id = self.samples[idx]
        # Parse the raw CSV file.
        data = parse_log(file_path)
        if self.debug:
            print(f"[DEBUG] Processing file: {file_path}")
            print(f"[DEBUG] Raw mouse events: {len(data['mouse'])}, Raw keyboard events: {len(data['keyboard'])}")
        
        # Apply feature engineering.
        mouse_events = compute_mouse_velocity(data["mouse"])
        keyboard_events = compute_keyboard_hold_times(data["keyboard"])
        
        # Convert events into feature vectors.
        # For mouse: [timestamp, x, y, vx, vy]
        mouse_feats = []
        for e in mouse_events:
            if e["x"] is not None and e["y"] is not None:
                mouse_feats.append([e["timestamp"], e["x"], e["y"], e["vx"], e["vy"]])
        # For keyboard: [timestamp, key (as numeric), action (1 for press, 0 for release), hold_time]
        key_feats = []
        for e in keyboard_events:
            # Convert key character to its ASCII value. If conversion fails, use 0.
            try:
                key_val = float(ord(e["key"][0])) if e["key"] else 0.0
            except Exception:
                key_val = 0.0
            action_val = 1.0 if e["action"] == "press" else 0.0
            hold_time = e.get("hold_time", 0.0)
            key_feats.append([e["timestamp"], key_val, action_val, hold_time])
        
        # Normalize timestamps relative to the first event.
        if len(mouse_feats) > 0:
            start_time = mouse_feats[0][0]
            mouse_feats = [[x - start_time if i == 0 else x for i, x in enumerate(feat)] for feat in mouse_feats]
        if len(key_feats) > 0:
            start_time = key_feats[0][0]
            key_feats = [[x - start_time if i == 0 else x for i, x in enumerate(feat)] for feat in key_feats]
        
        # Pad sequences to fixed lengths.
        mouse_arr = pad_sequence(mouse_feats, self.max_mouse_len, feature_dim=5)
        key_arr = pad_sequence(key_feats, self.max_key_len, feature_dim=4)

        if self.debug:
            print(f"[DEBUG] After processing: ")
            print(f"         Mouse sequence length (before pad): {len(mouse_feats)} -> shape after pad: {mouse_arr.shape}")
            print(f"         Keyboard sequence length (before pad): {len(key_feats)} -> shape after pad: {key_arr.shape}")
        
        # Convert arrays to torch tensors.
        mouse_tensor = torch.tensor(mouse_arr, dtype=torch.float32)
        key_tensor = torch.tensor(key_arr, dtype=torch.float32)

        sample = {"mouse": mouse_tensor, "keyboard": key_tensor, "player_id": player_id, "session_id": session_id}
        if self.debug:
            print(f"[DEBUG] Sample player_id: {player_id}, session_id: {session_id}")
        return sample

if __name__ == "__main__":
    # For debugging purposes. Run this module to check the output of the dataset.
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/raw", help="Directory containing raw CSV session files")
    parser.add_argument("--debug", action="store_true", help="Enable debug printing")
    args = parser.parse_args()
    
    dataset = SessionDataset(data_dir=args.data_dir, debug=args.debug)
    print(f"Total samples in dataset: {len(dataset)}")
    sample = dataset[5]
    print("[DEBUG] Sample keys:", sample.keys())
    print("[DEBUG] Mouse tensor shape:", sample["mouse"].shape)
    print("[DEBUG] Keyboard tensor shape:", sample["keyboard"].shape)
    print("[DEBUG] Player ID:", sample["player_id"])
    print("[DEBUG] Session ID:", sample["session_id"])

    print("[DEBUG] First 5 rows of Mouse tensor:")
    print(sample["mouse"][:5])

    print("[DEBUG] First 5 rows of Keyboard tensor:")
    print(sample["keyboard"][:5])
