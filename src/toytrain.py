import argparse
import pickle
import time
import math
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader
import yaml

from model import KeystrokeTransformer
from metrics import Metric

# ─── Config ─────────────────────────────────────────────────────────────────────
CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "config.yaml"
with open(CONFIG_PATH, "r") as f:
    cfg = yaml.safe_load(f)

# ─── Device ─────────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True

class ToyTripletDataset(Dataset):
    """Samples triplets: anchor+positive from user0, negative from user1"""
    def __init__(self, users, batch_size, epoch_batches, seq_len):
        assert len(users) >= 2, "Require at least two users for negative sampling"
        self.users   = users
        self.bs      = batch_size
        self.eb      = epoch_batches
        self.seq_len = seq_len

    def __len__(self):
        return self.bs * self.eb

    def __getitem__(self, idx):
        # choose genuine user 0, imposter user 1
        gu, iu = 0, 1
        # select two distinct sessions for anchor/positive
        g1, g2 = np.random.choice(len(self.users[gu]), 2, replace=False)
        # negative from second user
        ni = np.random.randint(len(self.users[iu]))

        def pad(seq):
            arr = np.array(seq, dtype=np.float32)
            L, D = arr.shape
            if L < self.seq_len:
                pad = np.zeros((self.seq_len - L, D), dtype=np.float32)
                arr = np.vstack([arr, pad])
            else:
                arr = arr[:self.seq_len]
            return torch.from_numpy(arr)

        A = pad(self.users[gu][g1])
        P = pad(self.users[gu][g2])
        N = pad(self.users[iu][ni])
        return A, P, N

# ─── Feature extraction ──────────────────────────────────────────────────────────
def extract_features(dataset):
    """Append timing features (10 dims) to raw [T×3] sequences."""
    for user in dataset:
        for i, seq in enumerate(user):
            arr = np.array(seq, dtype=np.float32)
            arr = np.concatenate([arr, np.zeros((len(arr),7), dtype=np.float32)], axis=1)
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
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()

    # Load original training data
    full_train = pickle.load(open("data/processed/training_data.pickle", "rb"))
    # Select first two users
    toy_users = [ full_train[0], full_train[1] ]

    # Feature extraction
    extract_features(toy_users)

    # Config values
    bs     = cfg["hyperparams"]["batch_size"]["aalto"]
    eb     = cfg["hyperparams"]["epoch_batch_count"]["aalto"]
    sl     = cfg["data"]["keystroke_sequence_len"]
    fc     = cfg["hyperparams"]["keystroke_feature_count"]["aalto"]
    tl     = cfg["hyperparams"]["target_len"]
    lr     = cfg["hyperparams"]["learning_rate"]
    margin = cfg.get("training", {}).get("margin", 0.5)

    # Override for toy
    bs_toy = min(bs, 16)
    eb_toy = min(eb, 10)
    print(f"[Toytrain] bs={bs_toy}, eb={eb_toy}, sl={sl}, margin={margin}")

    # DataLoader
    train_dl = DataLoader(
        ToyTripletDataset(toy_users, bs_toy, eb_toy, sl),
        batch_size=bs_toy, shuffle=True
    )

    # Model, loss, optimizer
    model   = KeystrokeTransformer(6, fc, 20, 5, 10, sl, tl, 0.1).to(device)
    loss_fn = torch.nn.TripletMarginLoss(margin=margin)
    optim   = torch.optim.Adam(model.parameters(), lr=lr)

    # Training loop with distance logging
    for ep in range(1, args.epochs+1):
        t0, total_loss = time.time(), 0.0
        model.train()
        dpos_list, dneg_list = [], []
        for A,P,N in train_dl:
            A,P,N = A.to(device),P.to(device),N.to(device)
            optim.zero_grad()
            outA = model(A)
            outP = model(P)
            outN = model(N)
            dpos = (outA - outP).norm(dim=1).mean().item()
            dneg = (outA - outN).norm(dim=1).mean().item()
            dpos_list.append(dpos)
            dneg_list.append(dneg)
            loss = loss_fn(outA, outP, outN)
            loss.backward()
            optim.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_dl)
        print(f"Epoch {ep}/{args.epochs}  Loss={avg_loss:.4f}  d_pos={np.mean(dpos_list):.3f}  d_neg={np.mean(dneg_list):.3f}  {time.time()-t0:.2f}s")

    # Compute toy EER
    model.eval()
    with torch.no_grad():
        # build score arrays
        enr = []
        imp = []
        # embed enrollment (first two samples)
        for s in range(2):
            x = torch.from_numpy(pad := np.pad(toy_users[0][s], ((0,max(0,sl-len(toy_users[0][s]))),(0,0)), 'constant')).unsqueeze(0).to(device)
            enr.append(model(x))
        enr = torch.cat(enr,0)
        # embed verification: positive and negatives
        # positive: other sessions of user0
        for s in toy_users[0][2:5]:
            x = torch.from_numpy(pad := np.pad(s, ((0,max(0,sl-len(s))),(0,0)), 'constant')).unsqueeze(0).to(device)
            imp.append(model(x))
        # negative: few sessions of user1
        for s in toy_users[1][:3]:
            x = torch.from_numpy(pad := np.pad(s, ((0,max(0,sl-len(s))),(0,0)), 'constant')).unsqueeze(0).to(device)
            imp.append(model(x))
        scores_pos = torch.norm(enr.mean(0,keepdim=True) - torch.cat(imp[:3]), dim=1)
        scores_neg = torch.norm(enr.mean(0,keepdim=True) - torch.cat(imp[3:]), dim=1)
        eer, thr = Metric.eer_compute(scores_pos, scores_neg)
    print(f"Toy EER={eer:.2f}%  thr={thr:.4f}")
