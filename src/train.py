import argparse
import subprocess
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
    """Run preprocess.py to generate train/val/test pickles."""
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

        def pad_and_tensor(seq):
            arr = np.array(seq, dtype=np.float32)
            L, D = arr.shape
            if L < self.seq_len:
                pad = np.zeros((self.seq_len - L, D), dtype=np.float32)
                arr = np.vstack([arr, pad])
            elif L > self.seq_len:
                arr = arr[:self.seq_len]
            return torch.from_numpy(arr)

        A = pad_and_tensor(self.data[gu][g1])
        P = pad_and_tensor(self.data[gu][g2])
        N = pad_and_tensor(self.data[iu][ni])
        return A, P, N

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
        arr = np.array(self.users[user_idx][session_idx], dtype=np.float32)
        L, D = arr.shape
        if L < self.seq_len:
            pad = np.zeros((self.seq_len - L, D), dtype=np.float32)
            arr = np.vstack([arr, pad])
        elif L > self.seq_len:
            arr = arr[:self.seq_len]
        return torch.from_numpy(arr)

class TripletLoss(torch.nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, a, p, n):
        dp = (a - p).pow(2).sum(dim=1).sqrt()
        dn = (a - n).pow(2).sum(dim=1).sqrt()
        return torch.relu(dp - dn + self.margin).mean()

def extract_features(dataset):
    """Append timing features (10 dims) to each [T×3] sequence."""
    for user in dataset:
        for i, seq in enumerate(user):
            arr = np.array(seq, dtype=np.float32)
            arr = np.concatenate([arr, np.zeros((len(arr), 7), dtype=np.float32)], axis=1)
            for j in range(len(arr)):
                m    = arr[j,1] - arr[j,0]
                ud   = arr[j+1,0] - arr[j,1] if j+1 < len(arr) else 0.0
                dd   = arr[j+1,0] - arr[j,0] if j+1 < len(arr) else 0.0
                uu   = arr[j+1,1] - arr[j,1] if j+1 < len(arr) else 0.0
                du   = arr[j+1,1] - arr[j,0] if j+1 < len(arr) else 0.0
                t_ud = arr[j+2,0] - arr[j,1] if j+2 < len(arr) else 0.0
                t_dd = arr[j+2,0] - arr[j,0] if j+2 < len(arr) else 0.0
                t_uu = arr[j+2,1] - arr[j,1] if j+2 < len(arr) else 0.0
                t_du = arr[j+2,1] - arr[j,0] if j+2 < len(arr) else 0.0
                code = arr[j,2]
                arr[j,0] = m    /1000
                arr[j,1] = ud   /1000
                arr[j,2] = dd   /1000
                arr[j,3] = uu   /1000
                arr[j,4] = du   /1000
                arr[j,5] = t_ud /1000
                arr[j,6] = t_dd /1000
                arr[j,7] = t_uu /1000
                arr[j,8] = t_du /1000
                arr[j,9] = code /255
            user[i] = arr

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("epochs", type=int, help="number of epochs")
    parser.add_argument("--resume", help="checkpoint to resume from")
    args = parser.parse_args()

    # 1) Preprocess
    #preprocess()

    # 2) Load data
    with open("data/processed/training_data.pickle","rb") as f:
        train_data = pickle.load(f)
    with open("data/processed/validation_data.pickle","rb") as f:
        val_data   = pickle.load(f)

    # 3) Feature extraction
    extract_features(train_data)
    extract_features(val_data)

    # 4) Hyperparams
    bs = cfg["hyperparams"]["batch_size"]["aalto"]
    eb = cfg["hyperparams"]["epoch_batch_count"]["aalto"]
    sl = cfg["data"]["keystroke_sequence_len"]
    fc = cfg["hyperparams"]["keystroke_feature_count"]["aalto"]
    tl = cfg["hyperparams"]["target_len"]
    lr = cfg["hyperparams"]["learning_rate"]
    ne = cfg["hyperparams"]["number_of_enrollment_sessions"]["aalto"]
    nv = cfg["hyperparams"]["number_of_verify_sessions"]["aalto"]

    # 5) Dataloaders
    train_dl = DataLoader(TrainDataset(train_data, bs, eb, sl), batch_size=bs, shuffle=True)
    val_dl   = DataLoader(ValidationDataset(val_data, sl), batch_size=bs, shuffle=False)

    # 6) Model, loss, optimizer
    model   = KeystrokeTransformer(6, fc, 20, 5, 10, sl, tl, 0.1).to(device)
    loss_fn = TripletLoss()
    optim   = torch.optim.Adam(model.parameters(), lr=lr)

    best_eer = math.inf
    start_ep = 1

    # 7) Resume if requested
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(    ckpt["model_state_dict"])
        optim.load_state_dict(    ckpt["optimizer_state_dict"])
        best_eer = ckpt.get("best_eer", best_eer)
        start_ep = ckpt.get("epoch", start_ep-1) + 1
        print(f"Resumed from epoch {start_ep-1}, best EER={best_eer:.2f}%")

    out_dir = Path("best_models")
    out_dir.mkdir(exist_ok=True)

    # 8) Training loop
    for ep in range(start_ep, args.epochs+1):
        t0 = time.time()
        model.train()
        total_loss = 0.0

        for A, P, N in train_dl:
            A, P, N = A.to(device), P.to(device), N.to(device)
            optim.zero_grad()
            loss = loss_fn(model(A), model(P), model(N))
            loss.backward()
            optim.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_dl)

        # 9) Validation
        model.eval()
        embs = []
        with torch.no_grad():
            for batch in val_dl:
                embs.append(model(batch.to(device)))
        sessions = len(val_data[0])
        emb = torch.cat(embs, 0).view(len(val_data), sessions, tl)
        eer, _ = Metric.cal_user_eer_aalto(emb, ne, nv)
        dt = time.time() - t0

        print(f"Epoch {ep}/{args.epochs}  Loss={avg_loss:.4f}  EER={eer:.2f}%  {dt:.1f}s")

        if eer < best_eer:
            best_eer = eer
            ckpt_path = out_dir / f"checkpoint_ep{ep}_eer{eer:.2f}.pth"
            torch.save({
                "epoch":                ep,
                "model_state_dict":     model.state_dict(),
                "optimizer_state_dict": optim.state_dict(),
                "best_eer":             best_eer
            }, ckpt_path)
            print(f" ➔ saved best ({best_eer:.2f}%) at {ckpt_path}")
