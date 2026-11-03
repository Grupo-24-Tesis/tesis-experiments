import math

IOU_THRESHOLD = 0.5
SIZE_BUCKETS = [(0, 16, "<16"), (16, 24, "16-24"), (24, 32, "24-32"),
                (32, 64, "32-64"), (64, math.inf, ">=64")]


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def face_size(box) -> float:
    return min(box[2] - box[0], box[3] - box[1])


def size_bucket(box) -> str:
    s = face_size(box)
    for lo, hi, name in SIZE_BUCKETS:
        if lo <= s < hi:
            return name
    return SIZE_BUCKETS[-1][2]


def in_scope(gt, min_px=0) -> bool:
    return (not gt.get("ignore", False)
            and gt.get("occlusion") != "total"
            and face_size(gt["box"]) >= min_px)


def match_frame(dets, gts, min_px=0, iou_thr=IOU_THRESHOLD, frame_idx=None):
    scoped = [in_scope(g, min_px) for g in gts]
    taken = [False] * len(gts)
    rows = []
    for det in sorted(dets, key=lambda d: d["score"], reverse=True):
        best, best_iou = None, 0.0
        for i, gt in enumerate(gts):
            if scoped[i] and not taken[i]:
                o = iou(det["box"], gt["box"])
                if o > best_iou:
                    best, best_iou = i, o
        if best is not None and best_iou >= iou_thr:
            taken[best] = True
            gt = gts[best]
            rows.append(_row(frame_idx, "TP", det, gt, best_iou))
            continue
        hits_ignored = max((iou(det["box"], g["box"]) for i, g in enumerate(gts) if not scoped[i]),
                           default=0.0)
        outcome = "IGNORED" if hits_ignored >= iou_thr else "FP"
        rows.append(_row(frame_idx, outcome, det, None, max(best_iou, hits_ignored)))
    for i, gt in enumerate(gts):
        if not taken[i]:
            rows.append(_row(frame_idx, "FN" if scoped[i] else "IGNORED", None, gt, 0.0))
    return rows


def _row(frame_idx, outcome, det, gt, overlap):
    box = gt["box"] if gt is not None else det["box"]
    return {
        "frame_idx": frame_idx,
        "outcome": outcome,
        "score": det["score"] if det is not None else None,
        "iou": round(overlap, 4),
        "det_box": det["box"] if det is not None else None,
        "gt_box": gt["box"] if gt is not None else None,
        "identity": gt.get("identity") if gt is not None else None,
        "pose": gt.get("pose") if gt is not None else None,
        "occlusion": gt.get("occlusion") if gt is not None else None,
        "size_px": round(face_size(box), 1),
        "size_bucket": size_bucket(box),
    }


def prf(rows, min_score=None):
    tp = fp = fn = 0
    for r in rows:
        kept = r["score"] is not None and (min_score is None or r["score"] >= min_score)
        if r["outcome"] == "TP":
            if kept:
                tp += 1
            else:
                fn += 1
        elif r["outcome"] == "FP" and kept:
            fp += 1
        elif r["outcome"] == "FN":
            fn += 1
    p = tp / (tp + fp) if tp + fp else 0.0
    r_ = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r_ / (p + r_) if p + r_ else 0.0
    return {"precision": p, "recall": r_, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def threshold_sweep(rows, thresholds):
    curve = [{"threshold": t, **prf(rows, t)} for t in thresholds]
    best = max(curve, key=lambda c: (c["f1"], c["threshold"]))
    return curve, best


def average_precision(rows) -> float:
    scored = sorted((r for r in rows if r["outcome"] in ("TP", "FP")),
                    key=lambda r: r["score"], reverse=True)
    n_pos = sum(1 for r in rows if r["outcome"] in ("TP", "FN"))
    if n_pos == 0:
        return 0.0
    tp = fp = 0
    precisions, recalls = [], []
    for r in scored:
        if r["outcome"] == "TP":
            tp += 1
        else:
            fp += 1
        precisions.append(tp / (tp + fp))
        recalls.append(tp / n_pos)
    ap, prev_r = 0.0, 0.0
    for i, rec in enumerate(recalls):
        ap += (rec - prev_r) * max(precisions[i:])
        prev_r = rec
    return ap


def summarize(rows, min_score=None, prefix="det."):
    out = {prefix + k: v for k, v in prf(rows, min_score).items()}
    out[prefix + "ap50"] = average_precision(rows)
    for _, _, name in SIZE_BUCKETS:
        sub = [r for r in rows if r["size_bucket"] == name]
        out[f"{prefix}f1@{name}"] = prf(sub, min_score)["f1"] if sub else None
        out[f"{prefix}n_gt@{name}"] = sum(1 for r in sub if r["outcome"] in ("TP", "FN"))
    for key in ("pose", "occlusion"):
        values = sorted({r[key] for r in rows if r[key]})
        for v in values:
            sub = [r for r in rows if r[key] == v]
            out[f"{prefix}recall@{key}={v}"] = prf(sub, min_score)["recall"]
    return out
