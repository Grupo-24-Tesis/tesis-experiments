import math

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

MIN_GENUINE_GAP_S = 2.0
MIN_IMPOSTORS = 100


def build_pairs(identities, times_s, min_gap_s=MIN_GENUINE_GAP_S):
    genuine, impostor = [], []
    n = len(identities)
    for i in range(n):
        for j in range(i + 1, n):
            if identities[i] == identities[j]:
                if abs(times_s[i] - times_s[j]) >= min_gap_s:
                    genuine.append((i, j))
            else:
                impostor.append((i, j))
    return np.array(genuine, dtype=int).reshape(-1, 2), np.array(impostor, dtype=int).reshape(-1, 2)


def l2_normalize(emb):
    emb = np.asarray(emb, dtype=np.float64)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    return emb / np.where(norms == 0, 1, norms)


def pair_distances(emb, pairs, metric="cosine"):
    if len(pairs) == 0:
        return np.array([])
    e = l2_normalize(emb)
    a, b = e[pairs[:, 0]], e[pairs[:, 1]]
    if metric == "cosine":
        return 1.0 - np.sum(a * b, axis=1)
    if metric == "euclidean_l2":
        return np.linalg.norm(a - b, axis=1)
    raise ValueError(f"métrica desconocida: {metric}")


def tar_at_far(genuine_d, impostor_d, far):
    imp = np.sort(np.asarray(impostor_d))
    gen = np.asarray(genuine_d)
    if len(imp) == 0 or len(gen) == 0:
        return {"tar": None, "tau": None, "far_actual": None}
    allowed = math.floor(far * len(imp) + 1e-9)
    if allowed == 0:
        tau = np.nextafter(imp[0], -np.inf)
    else:
        tau = imp[allowed - 1]
        if np.count_nonzero(imp <= tau) > allowed:
            tau = np.nextafter(tau, -np.inf)
    return {"tar": float(np.mean(gen <= tau)), "tau": float(tau),
            "far_actual": float(np.mean(imp <= tau))}


def eer(genuine_d, impostor_d):
    y = np.r_[np.ones(len(genuine_d)), np.zeros(len(impostor_d))]
    fpr, tpr, _ = roc_curve(y, -np.r_[genuine_d, impostor_d])
    fnr = 1 - tpr
    i = int(np.nanargmin(np.abs(fnr - fpr)))
    return float((fpr[i] + fnr[i]) / 2)


def summarize(genuine_d, impostor_d, prefix="emb."):
    gen, imp = np.asarray(genuine_d), np.asarray(impostor_d)
    out = {prefix + "n_genuine": int(len(gen)), prefix + "n_impostor": int(len(imp))}
    if len(gen) == 0 or len(imp) == 0:
        return out
    for far, name in ((0.01, "tar@far0.01"), (0.001, "tar@far0.001")):
        res = tar_at_far(gen, imp, far)
        out[prefix + name] = res["tar"]
        out[prefix + name.replace("tar", "tau")] = res["tau"]
    y = np.r_[np.ones(len(gen)), np.zeros(len(imp))]
    out[prefix + "auc"] = float(roc_auc_score(y, -np.r_[gen, imp]))
    out[prefix + "eer"] = eer(gen, imp)
    out[prefix + "intra_mean"], out[prefix + "intra_std"] = float(gen.mean()), float(gen.std())
    out[prefix + "inter_mean"], out[prefix + "inter_std"] = float(imp.mean()), float(imp.std())
    pooled = math.sqrt((gen.var() + imp.var()) / 2)
    out[prefix + "dprime"] = float(abs(imp.mean() - gen.mean()) / pooled) if pooled > 0 else None
    out[prefix + "weak_far0.01"] = bool(len(imp) < MIN_IMPOSTORS)
    return out
