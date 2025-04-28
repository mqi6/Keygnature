import subprocess
import sys
import pickle
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import yaml

from model import KeystrokeTransformer
from metrics import Metric

# --- load YAML config ---
CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "config.yaml"
with open(CONFIG_PATH, "r") as f:
    cfg = yaml.safe_load(f)

def preprocess():
    """Unpack raw ZIP into data/processed and write pickles."""
    script = Path(__file__).parent / "preprocess.py"
    subprocess.run(f"python {script}", shell=True, check=True)

class TestDataset(Dataset):
    def __init__(self, data, seq_len):
        self.data    = data
        self.seq_len = seq_len
        # assume each user has same number of sessions
        self.sessions = len(data[0])

    def __len__(self):
        return len(self.data) * self.sessions

    def __getitem__(self, idx):
        user_idx = idx // self.sessions
        sess_idx = idx %  self.sessions
        seq = self.data[user_idx][sess_idx]
        return self._pad(seq)

    def _pad(self, seq):
        L, D = seq.shape
        if L == self.seq_len:
            return seq
        if L < self.seq_len:
            pad = np.zeros((self.seq_len - L, D))
            return np.vstack([seq, pad])
        return seq[:self.seq_len]

if __name__ == "__main__":
    # 1) preprocess raw → testing_data.pickle
    preprocess()

    # 2) load testing data
    with open("data/processed/testing_data.pickle", "rb") as f:
        test_data = pickle.load(f)

    # 3) feature‐extract (same as in train)
    def extract_features(dataset):
        for user in dataset:
            for i, seq in enumerate(user):
                # append 7 zero‐cols
                seq = np.concatenate([seq, np.zeros((len(seq), 7))], axis=1)
                for j in range(len(seq)):
                    m    = seq[j,1] - seq[j,0]
                    ud   = seq[j+1,0] - seq[j,1] if j+1 < len(seq) else 0.0
                    dd   = seq[j+1,0] - seq[j,0] if j+1 < len(seq) else 0.0
                    uu   = seq[j+1,1] - seq[j,1] if j+1 < len(seq) else 0.0
                    du   = seq[j+1,1] - seq[j,0] if j+1 < len(seq) else 0.0
                    t_ud = seq[j+2,0] - seq[j,1] if j+2 < len(seq) else 0.0
                    t_dd = seq[j+2,0] - seq[j,0] if j+2 < len(seq) else 0.0
                    t_uu = seq[j+2,1] - seq[j,1] if j+2 < len(seq) else 0.0
                    t_du = seq[j+2,1] - seq[j,0] if j+2 < len(seq) else 0.0
                    code = seq[j,2]
                    seq[j,0] = m    /1000
                    seq[j,1] = ud   /1000
                    seq[j,2] = dd   /1000
                    seq[j,3] = uu   /1000
                    seq[j,4] = du   /1000
                    seq[j,5] = t_ud /1000
                    seq[j,6] = t_dd /1000
                    seq[j,7] = t_uu /1000
                    seq[j,8] = t_du /1000
                    seq[j,9] = code /255
                user[i] = seq

    extract_features(test_data)

    # 4) build dataloader
    bs  = cfg["hyperparams"]["batch_size"]["aalto"]
    sl  = cfg["data"]["keystroke_sequence_len"]

    ds = TestDataset(test_data, sl)
    dl = DataLoader(ds, batch_size=bs)

    # 5) load model
    ne  = cfg["hyperparams"]["number_of_enrollment_sessions"]["aalto"]
    nv  = cfg["hyperparams"]["number_of_verify_sessions"]["aalto"]
    tl  = cfg["hyperparams"]["target_len"]
    ckpt = sys.argv[2]  # e.g. "epoch_50_eer_1.23.pt"

    model = KeystrokeTransformer(
        num_layers=6,
        d_model=cfg["hyperparams"]["keystroke_feature_count"]["aalto"],
        k=20, heads=5, _heads=10,
        seq_len=sl, trg_len=tl, dropout=0.1
    )
    model.load_state_dict(torch.load(f"best_models/{ckpt}", map_location="cpu"))
    model.eval()

    # 6) infer embeddings
    with torch.no_grad():
        feats = []
        for batch in dl:
            feats.append(model(batch.float()))
    feats = torch.cat(feats, 0).view(len(test_data), ds.sessions, tl)

    # 7) evaluate
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    metric = sys.argv[1]  # "basic", "det", or "pca"
    if metric == "basic":
        eer, thr = Metric.cal_user_eer_aalto(feats, ne, nv)
        pd.DataFrame([[eer, thr]], columns=["eer","threshold"]).to_csv(results_dir/"basic.csv", index=False)
    elif metric == "det":
        Metric.save_DET_curve(feats, ne, str(results_dir))
    elif metric == "pca":
        Metric.save_PCA_curve(feats, ne, str(results_dir))
    else:
        raise ValueError("metric must be one of: basic, det, pca")
