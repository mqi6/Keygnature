import subprocess
import sys
import time
import math
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader
import yaml

from model import KeystrokeTransformer
from metrics import Metric

# ─── Load config ────────────────────────────────────────────────────────────────
CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "config.yaml"
with open(CONFIG_PATH, "r") as f:
    cfg = yaml.safe_load(f)

# ─── Device setup ───────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True

def preprocess():
    """Run the preprocess script to generate train/val/test pickles."""
    script = Path(__file__).parent / "preprocess.py"
    subprocess.run(f"python {script}", shell=True, check=True)

class TrainDataset(Dataset):
    def __init__(self, data, batch_size, epoch_batches, seq_len):
        self.data    = data
        self.bs      = batch_size
        self.eb      = epoch_batches
        self.seq_len = seq_len

    def __len__(self):
        return self.bs * self.eb

    def __getitem__(self, idx):
        gu = np.random.randint(len(self.data))
        iu = np.random.choice([i for i in range(len(self.data)) if i != gu])
        g1, g2 = np.random.choice(len(self.data[gu]), 2, replace=False)
        ni     = np.random.randint(len(self.data[iu]))
        return (
            self._pad(self.data[gu][g1]),
            self._pad(self.data[gu][g2]),
            self._pad(self.data[iu][ni]),
        )

    def _pad(self, seq):
        L, D = seq.shape
        if L == self.seq_len:
            return seq
        if L < self.seq_len:
            pad = np.zeros((self.seq_len - L, D))
            return np.vstack([seq, pad])
        return seq[:self.seq_len]

class ValidationDataset(Dataset):
    def __init__(self, users, seq_len):
        self.users    = users
        self.seq_len  = seq_len
        self.sessions = len(users[0])

    def __len__(self):
        return len(self.users) * self.sessions

    def __getitem__(self, idx):
        user_idx    = idx // self.sessions
        session_idx = idx %  self.sessions
        seq         = self.users[user_idx][session_idx]
        L, D        = seq.shape
        if L == self.seq_len:
            return seq
        if L < self.seq_len:
            pad = np.zeros((self.seq_len - L, D))
            return np.vstack([seq, pad])
        return seq[:self.seq_len]

class TripletLoss(torch.nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, a, p, n):
        dp = (a - p).pow(2).sum(dim=1).sqrt()
        dn = (a - n).pow(2).sum(dim=1).sqrt()
        return torch.relu(dp - dn + self.margin).mean()

def extract_features(dataset):
    """Append timing features to each [T×3] sequence, turning it into [T×10]."""
    for user in dataset:
        for i, seq in enumerate(user):
            # append 7 zero columns for the new features
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
                seq[j,0] = m    / 1000
                seq[j,1] = ud   / 1000
                seq[j,2] = dd   / 1000
                seq[j,3] = uu   / 1000
                seq[j,4] = du   / 1000
                seq[j,5] = t_ud / 1000
                seq[j,6] = t_dd / 1000
                seq[j,7] = t_uu / 1000
                seq[j,8] = t_du / 1000
                seq[j,9] = code / 255
            user[i] = seq

if __name__ == "__main__":
    # 1) Preprocess → generate pickles
    #preprocess()

    # 2) Load pickles
    with open("data/processed/training_data.pickle",   "rb") as f:
        train_data = pickle.load(f)
    with open("data/processed/validation_data.pickle", "rb") as f:
        val_data   = pickle.load(f)

    # 3) Feature extraction
    extract_features(train_data)
    extract_features(val_data)

    # 4) Hyperparameters
    bs = cfg["hyperparams"]["batch_size"]["aalto"]
    eb = cfg["hyperparams"]["epoch_batch_count"]["aalto"]
    sl = cfg["data"]["keystroke_sequence_len"]
    fc = cfg["hyperparams"]["keystroke_feature_count"]["aalto"]
    tl = cfg["hyperparams"]["target_len"]
    lr = cfg["hyperparams"]["learning_rate"]
    ne = cfg["hyperparams"]["number_of_enrollment_sessions"]["aalto"]
    nv = cfg["hyperparams"]["number_of_verify_sessions"]["aalto"]

    # 5) DataLoaders
    train_ds = TrainDataset(train_data, bs, eb, sl)
    train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True)
    val_ds   = ValidationDataset(val_data, sl)
    val_dl   = DataLoader(val_ds, batch_size=bs, shuffle=False)

    # 6) Model, loss, optimizer
    model   = KeystrokeTransformer(
        num_layers=6,
        d_model=fc,
        k=20,
        heads=5,
        _heads=10,
        seq_len=sl,
        trg_len=tl,
        dropout=0.1
    ).to(device)
    loss_fn = TripletLoss()
    optim   = torch.optim.Adam(model.parameters(), lr=lr)

    # 7) Training loop
    best_eer = math.inf
    out_dir  = Path(__file__).parent.parent / "best_models"
    out_dir.mkdir(exist_ok=True)
    epochs = int(sys.argv[1])

    for ep in range(1, epochs+1):
        start = time.time()
        model.train()
        total_loss = 0.0
        for A, P, N in train_dl:
            # move batch to GPU/CPU
            A, P, N = A.to(device), P.to(device), N.to(device)
            optim.zero_grad()
            loss = loss_fn(
                model(A.float()),
                model(P.float()),
                model(N.float())
            )
            loss.backward()
            optim.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_dl)

        # validation
        model.eval()
        with torch.no_grad():
            emb_batches = []
            for batch in val_dl:
                batch = batch.to(device)
                emb_batches.append(model(batch.float()))
        emb = torch.cat(emb_batches, 0).view(len(val_data), 15, tl)
        eer, _ = Metric.cal_user_eer_aalto(emb, ne, nv)
        elapsed = time.time() - start

        print(f"Epoch {ep}/{epochs}  Loss={avg_loss:.4f}  EER={eer:.2f}%  {elapsed:.1f}s")

        if eer < best_eer:
            best_eer = eer
            torch.save(
                model.state_dict(),
                out_dir / f"epoch_{ep}_eer_{eer:.2f}.pt"
            )
            print(f" ➔ saved best ({best_eer:.2f}%)")
