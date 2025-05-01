import torch
import numpy as np
import math
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

plt.style.use('seaborn-v0_8-bright')
plt.rcParams['axes.facecolor'] = 'white'
mpl.rcParams.update({"axes.grid": True, "grid.color": "black"})
mpl.rc('axes', edgecolor='black')
plt.rcParams.update({'font.size': 13})

class Metric:
    @staticmethod
    def eer_compute(scores_g, scores_i, eps: float = 1e-6):
        # concatenate genuine and imposter scores
        all_scores = torch.cat([scores_g, scores_i], dim=0)
        if all_scores.numel() == 0:
            return 50.0, 0.0
        ini = all_scores.min().item()
        fin = all_scores.max().item()
        # if no variation, return random-guess EER
        if abs(fin - ini) < eps:
            return 50.0, ini
        # compute thresholds
        far, frr, ths = [], [], []
        paso = (fin - ini) / 10000
        thr = ini - paso
        while thr < fin + paso:
            far.append((scores_i >= thr).float().mean().item())
            frr.append((scores_g <  thr).float().mean().item())
            ths.append(thr)
            thr += paso
        gap = torch.abs(torch.tensor(far) - torch.tensor(frr))
        idx = torch.argmin(gap, dim=0).item()
        eer = (far[idx] + frr[idx]) / 2 * 100
        return eer, ths[idx]

    @staticmethod
    def cal_user_eer_aalto(feats, num_enroll, num_verify):
        accs, ths = [], []
        U = feats.size(0)
        for i in range(U):
            enr  = feats[i, :num_enroll]
            rest = torch.cat([
                feats[i, num_enroll:],
                feats[:i, num_enroll:].reshape(-1, feats.size(-1)),
                feats[i+1:, num_enroll:].reshape(-1, feats.size(-1))
            ], dim=0)
            scores = torch.mean((rest.unsqueeze(1) - enr.unsqueeze(0))**2, dim=-1).sqrt().mean(1)
            eer, th = Metric.eer_compute(scores[:num_verify], scores[num_verify:])
            accs.append(eer)
            ths.append(th)
        # average across users
        return float(sum(accs) / len(accs)), float(sum(ths) / len(ths))

    @staticmethod
    def save_DET_curve(feats, num_enroll, out_path):
        U = feats.size(0)
        mins, maxs = math.inf, -math.inf
        for i in range(U):
            enr  = feats[i, :num_enroll]
            rest = torch.cat([
                feats[i, num_enroll:],
                feats[:i, num_enroll:].reshape(-1, feats.size(-1)),
                feats[i+1:, num_enroll:].reshape(-1, feats.size(-1))
            ], 0)
            sc   = torch.mean((rest.unsqueeze(1) - enr.unsqueeze(0))**2, dim=-1).sqrt().mean(1)
            mins = min(mins, sc.min().item())
            maxs = max(maxs, sc.max().item())
        # generate DET points
        steps = 10000
        paso  = (maxs - mins) / steps if maxs > mins else 1.0
        FAR, FRR = [], []
        for s in range(steps + 1):
            thr = mins + s * paso
            f, r = 0.0, 0.0
            for i in range(U):
                enr  = feats[i, :num_enroll]
                rest = torch.cat([
                    feats[i, num_enroll:],
                    feats[:i, num_enroll:].reshape(-1, feats.size(-1)),
                    feats[i+1:, num_enroll:].reshape(-1, feats.size(-1))
                ], 0)
                sc  = torch.mean((rest.unsqueeze(1) - enr.unsqueeze(0))**2, dim=-1).sqrt().mean(1)
                f  += (sc[num_enroll:] >= thr).float().mean().item()
                r  += (sc[:num_enroll]     <  thr).float().mean().item()
            FAR.append(f/U)
            FRR.append(r/U)
        pd.DataFrame({"FAR": FAR, "FRR": FRR}).to_csv(f"{out_path}/far-frr.csv", index=False)

    @staticmethod
    def save_PCA_curve(feats, num_enroll, out_path):
        U, S, D = feats.shape
        users = np.random.choice(U, min(10, U), False)
        data  = feats[users].reshape(-1, D).cpu().numpy()
        vals  = TSNE(n_iter=1000, perplexity=14).fit_transform(data)
        labels= np.repeat(np.arange(len(users)), S)
        score = silhouette_score(vals, labels)
        pd.DataFrame([[score]], columns=["Silhouette Score"]) \
          .to_csv(f"{out_path}/silhouette_score.csv", index=False)
        df = pd.DataFrame(vals, columns=["Dim1","Dim2"])
        df['User'] = labels
        g = sns.relplot(data=df, x="Dim1", y="Dim2", hue="User", height=6, aspect=1.2)
        g.ax.xaxis.grid(True, "minor", linewidth=0.25)
        g.ax.yaxis.grid(True, "minor", linewidth=0.25)
        g.despine(left=True, bottom=True)
        plt.savefig(f"{out_path}/pca_graph.png", dpi=300)
