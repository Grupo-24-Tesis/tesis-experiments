from collections import Counter, defaultdict

import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

NOISE = -1


def singleton_noise(pred):
    pred = list(pred)
    next_id = max([p for p in pred if p != NOISE], default=-1) + 1
    out = []
    for p in pred:
        if p == NOISE:
            out.append(next_id)
            next_id += 1
        else:
            out.append(p)
    return out


def purity(true, pred) -> float:
    groups = defaultdict(list)
    for t, p in zip(true, pred):
        groups[p].append(t)
    return sum(Counter(g).most_common(1)[0][1] for g in groups.values()) / len(true)


def bcubed(true, pred):
    by_pred, by_true = defaultdict(list), defaultdict(list)
    for i, (t, p) in enumerate(zip(true, pred)):
        by_pred[p].append(i)
        by_true[t].append(i)
    prec = rec = 0.0
    for t, p in zip(true, pred):
        same_cluster = by_pred[p]
        same_class = by_true[t]
        correct = sum(1 for j in same_cluster if true[j] == t)
        prec += correct / len(same_cluster)
        rec += correct / len(same_class)
    n = len(true)
    prec, rec = prec / n, rec / n
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return prec, rec, f1


def cluster_errors(true, pred):
    rows = []
    groups = defaultdict(list)
    for t, p in zip(true, pred):
        groups[p].append(t)
    for p, members in sorted(groups.items()):
        ids = Counter(members)
        kind = "noise" if p == NOISE else ("pure" if len(ids) == 1 else "merge")
        rows.append({"cluster": p, "kind": kind, "size": len(members),
                     "identities": dict(ids)})
    spread = defaultdict(set)
    for t, p in zip(true, pred):
        if p != NOISE:
            spread[t].add(p)
    for t, clusters in sorted(spread.items()):
        if len(clusters) > 1:
            rows.append({"cluster": None, "kind": "split", "size": len(clusters),
                         "identities": {t: sorted(clusters)}})
    return rows


def summarize(true, pred, prefix="clu."):
    true = list(true)
    raw = list(pred)
    pred = singleton_noise(raw)
    p, r, f1 = bcubed(true, pred)
    return {
        prefix + "purity": purity(true, pred),
        prefix + "inverse_purity": purity(pred, true),
        prefix + "ari": float(adjusted_rand_score(true, pred)),
        prefix + "nmi": float(normalized_mutual_info_score(true, pred)),
        prefix + "bcubed_p": p,
        prefix + "bcubed_r": r,
        prefix + "bcubed_f1": f1,
        prefix + "n_pred": len({x for x in raw if x != NOISE}),
        prefix + "n_true": len(set(true)),
        prefix + "noise_rate": float(np.mean([x == NOISE for x in raw])) if raw else 0.0,
    }
