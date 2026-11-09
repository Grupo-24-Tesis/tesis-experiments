import argparse
import csv
import glob
import json
import os
import traceback
from collections import Counter, defaultdict

import numpy as np

from harness import cluster as clu
from harness import items as items_mod
from harness import registry, runlog
from harness.metrics import clustering, detection, verification

GT_ROOT = runlog.GT_ROOT
SWEEP_THRESHOLDS = [round(0.5 + 0.01 * i, 2) for i in range(50)]


def load_gt(video_path):
    stem = os.path.splitext(os.path.basename(video_path))[0]
    d = os.path.join(GT_ROOT, stem)
    frames = runlog.read_json(os.path.join(d, "frames.json"))
    return {"dir": d, "frames": frames,
            "identities": runlog.read_json(runlog.IDENTITIES, {"identities": []}),
            "crops": runlog.read_json(os.path.join(d, "crop_labels.json"), {"items": {}})}


def reviewed_frames(gt):
    if not gt["frames"]:
        return {}
    return {int(k): f for k, f in gt["frames"]["frames"].items() if f.get("reviewed")}


def read_events(run_dir):
    path = os.path.join(run_dir, "events.jsonl")
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f]


def write_csv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        open(path, "w").close()
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: json.dumps(v) if isinstance(v, (list, dict)) else v for k, v in r.items()})


def eval_detect(run_dir, cfg, gt):
    m, missing = {}, []
    dets = runlog.read_json(os.path.join(run_dir, "outputs", "detections.json"), {})
    m["det.n_dets"] = sum(len(v) for v in dets.values())
    m["det.n_frames"] = len(dets)
    frames = reviewed_frames(gt)
    if not frames:
        return m, ["fotogramas GT sin revisar (GT-02)"]
    min_px = cfg["gt"]["min_face_px"]
    rows = []
    for idx, fr in sorted(frames.items()):
        rows += detection.match_frame(dets.get(str(idx), []), fr["faces"], min_px=min_px, frame_idx=idx)
    op = cfg["stage2"]["FACE_CONFIDENCE_THRESHOLD"] if cfg["detect"]["detector"] == "pipeline" else cfg["detect"]["threshold"]
    m["det.operating_threshold"] = op
    m.update(detection.summarize(rows, op))
    curve, best = detection.threshold_sweep(rows, SWEEP_THRESHOLDS)
    m["det.best_threshold"] = best["threshold"]
    m["det.best_f1"] = best["f1"]
    m["det.gt_frames_used"] = len(frames)
    m["det.gt_frames_total"] = len(gt["frames"]["frames"])
    write_csv(os.path.join(run_dir, "classification", "det_matches.csv"), rows)
    write_csv(os.path.join(run_dir, "classification", "pr_curve.csv"), curve)
    if len(frames) < len(gt["frames"]["frames"]):
        missing.append(f"solo {len(frames)}/{len(gt['frames']['frames'])} fotogramas GT revisados")
    return m, missing


def eval_pipeline(run_dir, cfg, gt):
    events = read_events(run_dir)
    m, missing = {}, []
    s1 = runlog.read_json(os.path.join(run_dir, "outputs", "candidates", "stage1_summary.json"), {})
    results = runlog.read_json(os.path.join(run_dir, "outputs", "faces", "results.json"), {"faces": []})
    m["trk.n_tracks"] = s1.get("n_tracks")
    m["trk.n_crops"] = s1.get("n_crops")
    m["trk.scene_cuts"] = sum(1 for e in events if e["event"] == "scene_cut")
    frames = [e for e in events if e["event"] == "frame"]
    m["trk.frames_analysed"] = len(frames)
    m["trk.yolo_ms_mean"] = float(np.mean([e["infer_ms"] for e in frames])) if frames else None
    cands = [e for e in events if e["event"] == "candidate"]
    reasons = Counter(e["reject_reason"] or "accepted" for e in cands)
    for k in ("accepted", "no_face", "below_threshold", "too_small", "crop_failed"):
        m[f"sel.candidates_{k}"] = reasons.get(k, 0)
    faces = results["faces"]
    m["sel.n_faces"] = len(faces)
    m["sel.face_rate"] = len(faces) / s1["n_tracks"] if s1.get("n_tracks") else None
    chosen = {(e["track_id"], e["chosen_crop"]) for e in events if e["event"] == "track_result" and e["chosen_crop"]}
    best = [e for e in cands if (e["track_id"], e["crop"]) in chosen]
    if best:
        m["sel.score_mean"] = float(np.mean([e["score"] for e in best]))
        m["sel.face_px_median"] = float(np.median([e["face_px"] for e in best]))
        m["sel.sharpness_median"] = float(np.median([e["sharpness"] for e in best]))
        m["trk.best_face_quality_median"] = float(np.median(
            [e["face_px"] * e["sharpness"] ** 0.5 for e in best]))
    rows = [{"track_id": e["track_id"], "crop": e["crop"], "frame_idx": e["frame_idx"],
             "chosen": (e["track_id"], e["crop"]) in chosen, "accepted": e["accepted"],
             "reject_reason": e["reject_reason"], "score": e["score"], "face_px": e["face_px"],
             "sharpness": e["sharpness"], "face_box_frame": e["face_box_frame"]} for e in cands]
    write_csv(os.path.join(run_dir, "classification", "candidates.csv"), rows)
    labels = {k: v for k, v in gt["crops"]["items"].items() if v.get("label")}
    if not labels:
        missing.append("no hay etiquetas de identidad de los rostros del pipeline (GT-03): "
                       "trk.coverage / trk.purity / sel.valid_rate / e2e.* sin calcular")
        return m, missing
    track_ids = defaultdict(list)
    unlabeled = 0
    for e in cands:
        if not e["accepted"]:
            continue
        lab = items_mod.crop_label(labels, e["frame_idx"], e["face_box_frame"])
        if lab is None:
            unlabeled += 1
        else:
            track_ids[e["track_id"]].append(lab)
    real = {t: [x for x in labs if x not in NON_PEOPLE] for t, labs in track_ids.items()}
    majority = {t: Counter(labs).most_common(1)[0][0] for t, labs in real.items() if labs}
    people = gt_people(gt, cfg)
    covered = set(majority.values()) & people if people else set(majority.values())
    m["trk.coverage"] = len(covered) / len(people) if people else None
    m["trk.fragmentation"] = (len(majority) / len(covered)) if covered else None
    m["trk.purity"] = float(np.mean([Counter(l).most_common(1)[0][1] / len(l) for l in real.values() if l])) if majority else None
    m["trk.id_switches"] = sum(1 for l in real.values() if len(set(l)) > 1)
    m["trk.faces_unlabeled"] = unlabeled
    sel_rows = []
    for e in best:
        lab = items_mod.crop_label(labels, e["frame_idx"], e["face_box_frame"])
        maj = majority.get(e["track_id"])
        cls = ("unlabeled" if lab is None else "not_a_face" if lab == "not_a_face" else
               "unknown" if lab == "unknown" else "valid" if lab == maj else "wrong_person")
        sel_rows.append({"track_id": e["track_id"], "crop": e["crop"], "label": lab,
                         "track_majority": maj, "class": cls})
    write_csv(os.path.join(run_dir, "classification", "selection.csv"), sel_rows)
    judged = [r for r in sel_rows if r["class"] != "unlabeled"]
    m["sel.valid_rate"] = sum(r["class"] == "valid" for r in judged) / len(judged) if judged else None
    out_people = [r["label"] for r in sel_rows if r["class"] != "unlabeled"]
    m.update(e2e_metrics(out_people, people, prefix="e2e.tracks_"))
    if unlabeled:
        missing.append(f"{unlabeled} rostros aceptados todavía sin etiquetar (GT-03)")
    return m, missing


NON_PEOPLE = {"not_a_face", "unknown"}


def gt_people(gt, cfg):
    min_px = cfg["gt"]["min_face_px"]
    return {f["identity"] for fr in reviewed_frames(gt).values() for f in fr["faces"]
            if f.get("identity") and detection.in_scope(f, min_px)}


def e2e_metrics(output_labels, people, prefix="e2e."):
    real = [x for x in output_labels if x and x not in NON_PEOPLE]
    covered = set(real) & people if people else set(real)
    rows = []
    for p in sorted(people):
        n = real.count(p)
        rows.append({"identity": p, "outputs": n,
                     "class": "missed" if n == 0 else "found_once" if n == 1 else "found_duplicated"})
    return {prefix + "identity_recall": len(covered) / len(people) if people else None,
            prefix + "duplicates": (len(real) - len(covered)) / len(covered) if covered else None,
            prefix + "false_outputs": sum(1 for x in output_labels if not x or x in NON_PEOPLE),
            prefix + "n_outputs": len(output_labels),
            "_people_rows": rows}


def load_embeddings(run_dir, cfg=None, gt=None):
    emb_dir = os.path.join(run_dir, "outputs", "embeddings")
    its = runlog.read_json(os.path.join(run_dir, "outputs", "items.json"), [])
    if cfg is not None and cfg["items"]["source"] == "pipeline":
        labels = gt["crops"]["items"]
        for it in its:
            it["identity"] = items_mod.crop_label(labels, it["frame_idx"], it["face_box_frame"])
    index = runlog.read_json(os.path.join(emb_dir, "embeddings_index.json"), [])
    mat = np.load(os.path.join(emb_dir, "embeddings.npy"))
    return its, index, mat


def labeled_mask(its, index):
    return np.array([ix["ok"] and bool(it.get("identity")) and it["identity"] not in NON_PEOPLE
                     for it, ix in zip(its, index)], dtype=bool)


def eval_embed_spread(run_dir, cfg, its, index, mat):
    ok = np.array([ix["ok"] for ix in index], dtype=bool)
    ctrl = np.array([bool(it.get("control")) for it in its], dtype=bool)
    d = clu.distance_matrix(mat[ok], cfg["embed"]["distance"])
    f_idx, c_idx = np.where(~ctrl[ok])[0], np.where(ctrl[ok])[0]
    ff = d[np.ix_(f_idx, f_idx)][np.triu_indices(len(f_idx), 1)]
    fc = d[np.ix_(f_idx, c_idx)].ravel()
    m = {"emb0.n_faces": int(len(f_idx)), "emb0.n_controls": int(len(c_idx)),
         "emb0.n_failed": int((~ok).sum()),
         "emb0.n_not_aligned": sum(1 for ix in index if ix["ok"] and not ix["aligned"])}
    if len(ff):
        m.update({"emb0.face_face_p10": float(np.percentile(ff, 10)),
                  "emb0.face_face_median": float(np.median(ff)),
                  "emb0.face_face_p90": float(np.percentile(ff, 90))})
    if len(fc):
        m["emb0.face_control_median"] = float(np.median(fc))
    if len(ff) and len(fc):
        m["emb0.control_gap"] = m["emb0.face_control_median"] - m["emb0.face_face_median"]
        m["emb0.collapsed"] = bool(m["emb0.face_face_median"] < 0.05)
    np.savez_compressed(os.path.join(run_dir, "outputs", "distances.npz"), genuine=ff, impostor=fc)
    return m, []


def eval_embed(run_dir, cfg, gt):
    its, index, mat = load_embeddings(run_dir, cfg, gt)
    if cfg["items"]["source"] == "auto":
        return eval_embed_spread(run_dir, cfg, its, index, mat)
    keep = labeled_mask(its, index)
    sub = [it for it, k in zip(its, keep) if k]
    emb = mat[keep]
    m = {"emb.n_items": len(its), "emb.n_used": int(keep.sum()),
         "emb.n_identities": len({it["identity"] for it in sub}),
         "emb.n_not_aligned": sum(1 for ix in index if ix["ok"] and not ix["aligned"]),
         "emb.n_failed": sum(1 for ix in index if not ix["ok"])}
    metric = cfg["embed"]["distance"]
    gen, imp = verification.build_pairs([it["identity"] for it in sub], [it["time_s"] for it in sub],
                                        cfg["gt"]["min_genuine_gap_s"])
    gd, idist = verification.pair_distances(emb, gen, metric), verification.pair_distances(emb, imp, metric)
    m.update(verification.summarize(gd, idist))
    for key, values in (("size_bucket", [b[2] for b in detection.SIZE_BUCKETS]),
                        ("pose", ["frontal", "semi"])):
        for v in values:
            def in_slice(it):
                return (detection.size_bucket(it["box"]) if key == "size_bucket" else it.get("pose")) == v
            sel = np.array([in_slice(it) for it in sub], dtype=bool)
            g = gen[sel[gen[:, 0]] & sel[gen[:, 1]]] if len(gen) else gen
            i = imp[sel[imp[:, 0]] & sel[imp[:, 1]]] if len(imp) else imp
            if len(g) and len(i):
                res = verification.tar_at_far(verification.pair_distances(emb, g, metric),
                                              verification.pair_distances(emb, i, metric), 0.01)
                m[f"emb.tar@far0.01@{key}={v}"] = res["tar"]
                m[f"emb.n_genuine@{key}={v}"] = int(len(g))
    tau = m.get("emb.tau@far0.01")
    pair_stats = defaultdict(list)
    ids = [it["identity"] for it in sub]
    for (a, b), dist in zip(imp, idist):
        pair_stats[tuple(sorted((ids[a], ids[b])))].append(dist)
    rows = [{"identity_a": a, "identity_b": b, "n_pairs": len(ds), "mean_distance": float(np.mean(ds)),
             "min_distance": float(np.min(ds)), "confusable": bool(tau is not None and np.min(ds) <= tau)}
            for (a, b), ds in sorted(pair_stats.items())]
    write_csv(os.path.join(run_dir, "classification", "pairs_summary.csv"), rows)
    np.savez_compressed(os.path.join(run_dir, "outputs", "distances.npz"), genuine=gd, impostor=idist)
    missing = [] if m["emb.n_used"] else ["no hay rostros etiquetados para calcular embeddings (GT-02)"]
    return m, missing


def cannot_link_pairs(cfg, its):
    run_dir = runlog.exp_path(cfg["items"]["pipeline_run"])
    together, offset, max_id = [], 0, 0
    with open(os.path.join(run_dir, "events.jsonl")) as f:
        for line in f:
            e = json.loads(line)
            if e["event"] == "scene_cut":
                offset = max_id
            elif e["event"] == "frame" and e.get("track_ids"):
                ids = {str(i + offset) for i in e["track_ids"]}
                max_id = max(max_id, *(i + offset for i in e["track_ids"]))
                together.append(ids)
    co = set()
    for ids in together:
        for a in ids:
            for b in ids:
                if a < b:
                    co.add((a, b))
    pairs = []
    for i, a in enumerate(its):
        for j in range(i + 1, len(its)):
            ta, tb = its[i]["track_id"], its[j]["track_id"]
            if ta != tb and tuple(sorted((ta, tb))) in co:
                pairs.append((i, j))
    return pairs


def eval_cluster(run_dir, cfg, gt):
    its, index, mat = load_embeddings(run_dir, cfg, gt)
    ok = np.array([ix["ok"] for ix in index], dtype=bool)
    all_its = [it for it, k in zip(its, ok) if k]
    d = clu.distance_matrix(mat[ok], cfg["embed"]["distance"])
    c = cfg["cluster"]
    if c["cannot_link"] and cfg["items"]["source"] == "pipeline":
        d = clu.apply_cannot_link(d, cannot_link_pairs(cfg, all_its))
    labels = clu.cluster(d, c["algo"], c[c["algo"]])
    lab = [it.get("identity") for it in all_its]
    scored = [i for i, x in enumerate(lab) if x and x not in NON_PEOPLE]
    m = {"clu.n_items": len(all_its), "clu.n_scored": len(scored)}
    missing = []
    if scored:
        m.update(clustering.summarize([lab[i] for i in scored], [int(labels[i]) for i in scored]))
        write_csv(os.path.join(run_dir, "classification", "cluster_errors.csv"),
                  clustering.cluster_errors([lab[i] for i in scored], [int(labels[i]) for i in scored]))
    else:
        missing.append("no hay rostros etiquetados (GT-02 / GT-03)")
    write_csv(os.path.join(run_dir, "outputs", "clusters.csv"),
              [{"item_id": it["item_id"], "identity": it.get("identity"), "cluster": int(k),
                "track_id": it.get("track_id"), "frame_idx": it.get("frame_idx")}
               for it, k in zip(all_its, labels)])
    if cfg["items"]["source"] == "pipeline":
        groups = defaultdict(list)
        for i, k in enumerate(clustering.singleton_noise([int(x) for x in labels])):
            groups[k].append(lab[i])
        majority = [Counter(g).most_common(1)[0][0] for g in groups.values()]
        e2e = e2e_metrics(majority, gt_people(gt, cfg))
        write_csv(os.path.join(run_dir, "classification", "e2e_people.csv"), e2e.pop("_people_rows"))
        m.update(e2e)
        tracks = pipeline_tracks(cfg, gt)
        groups = group_people(all_its, labels, tracks)
        pm, rows, ppl = people_metrics(groups, tracks, all_its, mat[ok], gt, cfg)
        unlabeled = sum(1 for r in rows if r["output_label"] is None)
        if unlabeled == len(rows):
            pm = {"ppl.n_outputs": len(rows)}
            missing.append("rostros de salida todavía sin etiquetar (GT-03): ppl.* sin calcular")
        elif unlabeled:
            missing.append(f"{unlabeled} de {len(rows)} rostros de salida todavía sin etiquetar (GT-03)")
        m.update(pm)
        write_csv(os.path.join(run_dir, "classification", "people.csv"), rows)
        runlog.write_json(os.path.join(run_dir, "outputs", "people.json"), ppl)
    return m, missing


def pipeline_tracks(cfg, gt):
    run_dir = runlog.exp_path(cfg["items"]["pipeline_run"])
    labels = gt["crops"]["items"]
    events = read_events(run_dir)
    by_track = defaultdict(list)
    for e in events:
        if e["event"] == "candidate":
            by_track[e["track_id"]].append(e)
    tracks = {}
    for e in events:
        if e["event"] != "track_result" or not e["chosen_crop"]:
            continue
        cands = by_track[e["track_id"]]
        best = next(c for c in cands if c["crop"] == e["chosen_crop"])
        labs = [items_mod.crop_label(labels, c["frame_idx"], c["face_box_frame"]) for c in cands if c["accepted"]]
        real = [x for x in labs if x and x not in NON_PEOPLE]
        tracks[e["track_id"]] = {
            "identity": Counter(real).most_common(1)[0][0] if real else None,
            "output_label": items_mod.crop_label(labels, best["frame_idx"], best["face_box_frame"]),
            "rank": tuple(best["rank_key"]), "frame_idx": best["frame_idx"]}
    return tracks


def group_people(all_its, labels, tracks):
    votes = defaultdict(list)
    for it, k in zip(all_its, labels):
        votes[it["track_id"]].append(int(k))
    key = {}
    for t in sorted(tracks, key=int):
        v = votes.get(t)
        if v:
            top = Counter(v).most_common()
            tied = {x for x, n in top if n == top[0][1]}
            key[t] = next(x for x in v if x in tied)
        else:
            key[t] = f"alone_{t}"
    groups = defaultdict(list)
    for t in sorted(tracks, key=int):
        groups[key[t]].append(t)
    return list(groups.values())


def people_metrics(groups, tracks, all_its, emb, gt, cfg):
    people = gt_people(gt, cfg)
    e = verification.l2_normalize(emb) if len(emb) else emb
    rows, ppl, out_labels = [], [], []
    for n, g in enumerate(groups):
        best = max(g, key=lambda t: tracks[t]["rank"])
        ids = Counter(tracks[t]["identity"] for t in g if tracks[t]["identity"])
        own = [i for i, it in enumerate(all_its) if it["track_id"] in g]
        centroid = None
        if own:
            c = e[own].mean(axis=0)
            centroid = (c / max(np.linalg.norm(c), 1e-12)).tolist()
        out = tracks[best]["output_label"]
        identity = ids.most_common(1)[0][0] if ids else (out if out and out not in NON_PEOPLE else None)
        rows.append({"person": n, "track_ids": " ".join(g), "n_tracks": len(g), "n_faces": len(own),
                     "track_identities": ", ".join(sorted(ids)), "identity": identity,
                     "output_track": best, "output_frame": tracks[best]["frame_idx"], "output_label": out,
                     "false_merge": len(ids) > 1})
        ppl.append({"person": n, "track_ids": g, "identity": identity, "output_label": out,
                    "false_merge": len(ids) > 1, "n_faces": len(own), "centroid": centroid})
        out_labels.append(out)
    m = e2e_metrics(out_labels, people, prefix="ppl.")
    m.pop("_people_rows")
    labeled = [r for r in rows if r["track_identities"]]
    m["ppl.false_merges"] = sum(r["false_merge"] for r in rows)
    m["ppl.merge_free"] = 1 - m["ppl.false_merges"] / len(labeled) if labeled else None
    per_id = Counter(t["identity"] for t in tracks.values() if t["identity"])
    multi = [i for i, n in per_id.items() if n > 1]
    joined = [i for i in multi if sum(1 for g in groups if any(tracks[t]["identity"] == i for t in g)) == 1]
    m["ppl.multi_track_identities"] = len(multi)
    m["ppl.multi_track_joined"] = len(joined) / len(multi) if multi else None
    covered = {x for x in out_labels if x and x not in NON_PEOPLE} & (people or set(out_labels))
    m["ppl.person_precision"] = len(covered) / len(out_labels) if out_labels else None
    r, p = m["ppl.identity_recall"], m["ppl.person_precision"]
    m["ppl.person_f1"] = 2 * p * r / (p + r) if p and r else 0.0
    found = {r["identity"] for r in rows if r["identity"]} & (people or {r["identity"] for r in rows if r["identity"]})
    m["ppl.found_recall"] = len(found) / len(people) if people else None
    m["ppl.groups_per_found_person"] = (sum(1 for r in rows if r["identity"] in found) / len(found)) if found else None
    m["ppl.unusable_output_faces"] = sum(1 for r in rows if r["identity"] and (not r["output_label"] or r["output_label"] in NON_PEOPLE))
    m["ppl.wrong_person_output_faces"] = sum(1 for r in rows if r["identity"] and r["output_label"]
                                             and r["output_label"] not in NON_PEOPLE and r["output_label"] != r["identity"])
    m["ppl.groups_without_person"] = sum(1 for r in rows if not r["identity"])
    return m, rows, ppl


XV_THRESHOLDS = [round(0.05 + 0.01 * i, 2) for i in range(91)]


SCORED = ("correct_match", "wrong_match", "missed_match", "correct_reject")


def xv_sources(cfg):
    meta = runlog.read_json(runlog.VIDEOS_META, {"videos": {}})["videos"]
    unsure = {p["id"] for p in runlog.read_json(runlog.IDENTITIES, {"identities": []})["identities"] if p.get("uncertain")}
    out = []
    for ref in cfg["xvideo"]["runs"]:
        d = runlog.exp_path(ref)
        ppl = runlog.read_json(os.path.join(d, "outputs", "people.json"))
        if ppl is None:
            raise RuntimeError(f"{ref} no tiene outputs/people.json: primero evalúe esa ejecución de agrupación")
        c = runlog.read_json(os.path.join(d, "config.json"))
        gt_src = load_gt(c["video"])
        its, index, mat = load_embeddings(d, c, gt_src)
        ok = np.array([ix["ok"] for ix in index], dtype=bool)
        stem = os.path.splitext(os.path.basename(c["video"]))[0]
        camera = meta.get(stem, {}).get("camera") or cfg["xvideo"].get("cameras", {}).get(stem, stem)
        people = [dict(p, uncertain=p["identity"] in unsure) for p in ppl if p["centroid"] is not None]
        out.append({"ref": ref, "video": stem, "camera": camera, "people": people, "unsure": unsure,
                    "faces": [it for it, k in zip(its, ok) if k],
                    "emb": verification.l2_normalize(mat[ok]),
                    "gt_frames": (gt_src["frames"] or {"frames": {}})["frames"],
                    "resolution": runlog.read_json(os.path.join(d, "run.json"), {}).get("video", {})})
    if len({s["video"] for s in out}) != len(out):
        raise ValueError("xvideo.runs debe venir de videos distintos")
    return out


XV_SLICE_PX = (0, 16, 20, 24)
POSE_MATCH_FRAMES = 8
POSE_MATCH_IOU = 0.3


def face_pose(gt_frames, frame_idx, box):
    best, best_iou = None, POSE_MATCH_IOU
    for k, fr in gt_frames.items():
        if not fr.get("reviewed") or abs(int(k) - frame_idx) > POSE_MATCH_FRAMES:
            continue
        for f in fr["faces"]:
            if f.get("ignore"):
                continue
            o = detection.iou(f["box"], box)
            if o > best_iou:
                best, best_iou = f.get("pose"), o
    return best


def face_size_px(it):
    b = it.get("face_box_frame") or it.get("box")
    return min(b[2] - b[0], b[3] - b[1]) if b else 0.0


def pair_auc(ea, ia, eb, ib, within):
    d, g = 1.0 - ea @ eb.T, ia[:, None] == ib[None, :]
    if within:
        iu = np.triu_indices(len(ia), 1)
        d, g = d[iu], g[iu]
    else:
        d, g = d.ravel(), g.ravel()
    auc = verification.summarize(d[g], d[~g], prefix="")["auc"] if g.any() and (~g).any() else None
    return auc, int(g.sum()), int((~g).sum())


def xv_face_slices(srcs):
    usable = []
    for s in srcs:
        keep = [i for i, it in enumerate(s["faces"]) if it.get("identity") and it["identity"] not in NON_PEOPLE
                and it["identity"] not in s["unsure"]]
        usable.append({
            "video": s["video"], "camera": s["camera"], "emb": s["emb"][keep],
            "ids": np.array([s["faces"][i]["identity"] for i in keep]),
            "px": np.array([face_size_px(s["faces"][i]) for i in keep]),
            "pose": np.array([face_pose(s["gt_frames"], s["faces"][i]["frame_idx"], s["faces"][i]["face_box_frame"]) or "unknown"
                              if s["faces"][i].get("face_box_frame") else "unknown" for i in keep])})
    px = np.concatenate([u["px"] for u in usable]) if usable else np.array([])
    stats = {"xv.face_px_median": float(np.median(px)) if len(px) else None,
             "xv.share_faces_lt13px": float(np.mean(px < 13)) if len(px) else None,
             "xv.share_faces_ge24px": float(np.mean(px >= 24)) if len(px) else None,
             "xv.n_faces_ge24px": int(np.sum(px >= 24))}
    rows = []
    for min_px in XV_SLICE_PX:
        for tag, pose_name, poses in (("any", "cualquier pose", None),
                                      ("frontal_semi", "frontal o semi", ("frontal", "semi"))):
            sel = [(u["px"] >= min_px) & (np.isin(u["pose"], poses) if poses else True) for u in usable]
            for i, u in enumerate(usable):
                auc, ng, ni = pair_auc(u["emb"][sel[i]], u["ids"][sel[i]], u["emb"][sel[i]], u["ids"][sel[i]], True)
                rows.append({"comparison": f"dentro de {u['video']}", "min_face_px": min_px, "pose": pose_name,
                             "n_faces": int(sel[i].sum()), "n_genuine": ng, "n_impostor": ni, "auc": auc})
                if auc is not None and poses is None and min_px == 0:
                    stats[f"xv.within.auc.{u['video']}"] = auc
            for a in range(len(usable)):
                for b in range(a + 1, len(usable)):
                    if usable[a]["camera"] != usable[b]["camera"]:
                        continue
                    ua, ub = usable[a], usable[b]
                    auc, ng, ni = pair_auc(ua["emb"][sel[a]], ua["ids"][sel[a]], ub["emb"][sel[b]], ub["ids"][sel[b]], False)
                    rows.append({"comparison": f"{ua['video']} frente a {ub['video']}", "min_face_px": min_px, "pose": pose_name,
                                 "n_faces": int(sel[a].sum() + sel[b].sum()), "n_genuine": ng, "n_impostor": ni, "auc": auc})
                    if auc is not None:
                        stats[f"xv.cross.auc.{tag}.px{min_px}"] = auc
                        stats[f"xv.cross.n_genuine.{tag}.px{min_px}"] = ng
    return rows, stats


def xv_open_set(srcs, t):
    rows = []
    for g in srcs:
        for q in srcs:
            if g is q:
                continue
            gal = g["people"]
            gal_ids = {p["identity"] for p in gal if p["identity"] and not p["uncertain"]}
            gal_unsure = any(p["uncertain"] for p in gal)
            gc = np.array([p["centroid"] for p in gal])
            same_camera = g["camera"] == q["camera"]
            for probe in q["people"]:
                if not probe["identity"] or probe["identity"] in NON_PEOPLE:
                    continue
                d = 1.0 - gc @ np.array(probe["centroid"]) if len(gal) else np.array([])
                order = np.argsort(d)
                near = gal[order[0]] if len(order) else None
                dmin = float(d[order[0]]) if len(order) else None
                present = probe["identity"] in gal_ids
                matched = near is not None and dmin < t
                if not same_camera or probe["uncertain"] or (matched and near["uncertain"]):
                    outcome = "unverified_match" if matched else "unverified_no_match"
                elif matched:
                    outcome = "correct_match" if near["identity"] == probe["identity"] else "wrong_match"
                elif present:
                    outcome = "missed_match"
                else:
                    outcome = "unverified_no_match" if gal_unsure else "correct_reject"
                rows.append({"gallery_video": g["video"], "probe_video": q["video"],
                             "same_camera": same_camera, "probe_person": probe["person"],
                             "identity": probe["identity"], "in_gallery": present,
                             "nearest_person": near["person"] if near else None,
                             "nearest_identity": near["identity"] if near else None, "distance": dmin,
                             "second_distance": float(d[order[1]]) if len(order) > 1 else None,
                             "threshold": t, "outcome": outcome})
    return rows


def xv_rates(rows, prefix):
    c = Counter(r["outcome"] for r in rows)
    out = {prefix + "unverified_matches": c.get("unverified_match", 0)}
    rows = [r for r in rows if r["outcome"] in SCORED]
    n = len(rows)
    out[prefix + "n_probes"] = n
    if not n:
        return out
    for k in SCORED:
        out[prefix + k] = c.get(k, 0)
    out[prefix + "accuracy"] = (c.get("correct_match", 0) + c.get("correct_reject", 0)) / n
    out[prefix + "wrong_match_rate"] = c.get("wrong_match", 0) / n
    return out


def eval_xvideo(run_dir, cfg, gt):
    srcs = xv_sources(cfg)
    m = {"xv.n_videos": len(srcs), "xv.n_people": sum(len(s["people"]) for s in srcs),
         "xv.n_labeled_people": sum(1 for s in srcs for p in s["people"] if p["identity"]),
         "xv.n_false_merged_people": sum(1 for s in srcs for p in s["people"] if p["false_merge"])}
    gen, imp, pair_rows = [], [], []
    for a in range(len(srcs)):
        for b in range(a + 1, len(srcs)):
            if srcs[a]["camera"] != srcs[b]["camera"]:
                continue
            for p in srcs[a]["people"]:
                for q in srcs[b]["people"]:
                    if not p["identity"] or not q["identity"] or p["uncertain"] or q["uncertain"]:
                        continue
                    d = float(1.0 - np.dot(p["centroid"], q["centroid"]))
                    same = p["identity"] == q["identity"]
                    (gen if same else imp).append(d)
                    pair_rows.append({"video_a": srcs[a]["video"], "person_a": p["person"], "identity_a": p["identity"],
                                      "video_b": srcs[b]["video"], "person_b": q["person"], "identity_b": q["identity"],
                                      "same_camera": srcs[a]["camera"] == srcs[b]["camera"],
                                      "genuine": same, "distance": d})
    write_csv(os.path.join(run_dir, "classification", "xv_person_pairs.csv"), pair_rows)
    m.update(verification.summarize(gen, imp, prefix="xv.person."))
    fgen, fimp = [], []
    for a in range(len(srcs)):
        for b in range(a + 1, len(srcs)):
            if srcs[a]["camera"] != srcs[b]["camera"]:
                continue
            ok_id = lambda it, s: it.get("identity") and it["identity"] not in NON_PEOPLE and it["identity"] not in s["unsure"]
            ia = [i for i, it in enumerate(srcs[a]["faces"]) if ok_id(it, srcs[a])]
            ib = [i for i, it in enumerate(srcs[b]["faces"]) if ok_id(it, srcs[b])]
            if not ia or not ib:
                continue
            d = 1.0 - srcs[a]["emb"][ia] @ srcs[b]["emb"][ib].T
            same = np.array([[srcs[a]["faces"][i]["identity"] == srcs[b]["faces"][j]["identity"] for j in ib] for i in ia])
            fgen += d[same].tolist()
            fimp += d[~same].tolist()
    m.update(verification.summarize(fgen, fimp, prefix="xv.face."))
    slice_rows, slice_stats = xv_face_slices(srcs)
    write_csv(os.path.join(run_dir, "classification", "xv_face_slices.csv"), slice_rows)
    m.update(slice_stats)
    for s in srcs:
        r = s["resolution"]
        m[f"xv.resolution.{s['video']}"] = f"{r.get('width')}x{r.get('height')} @ {round(r.get('fps') or 0, 1)} fps"
    sweep = []
    for t in XV_THRESHOLDS:
        rows = xv_open_set(srcs, t)
        r = {"threshold": t, **xv_rates(rows, ""),
             **xv_rates([x for x in rows if x["same_camera"]], "same_camera."),
             **xv_rates([x for x in rows if not x["same_camera"]], "cross_camera.")}
        r["person_tar"] = float(np.mean(np.array(gen) < t)) if gen else None
        r["person_far"] = float(np.mean(np.array(imp) < t)) if imp else None
        sweep.append(r)
    write_csv(os.path.join(run_dir, "classification", "xv_threshold_sweep.csv"), sweep)
    missing = []
    scored = [r for r in sweep if r.get("accuracy") is not None]
    if not scored:
        missing.append("no hay personas etiquetadas en las ejecuciones de origen (GT-03)")
        return m, missing
    if not gen:
        missing.append("ninguna persona aparece en ambos videos con esta configuración: la exactitud solo contaría rechazos, sin calcular")
        return m, missing
    top = max(r["accuracy"] for r in scored)
    best_ts = [r["threshold"] for r in scored if r["accuracy"] == top]
    runs, cur = [], [best_ts[0]]
    for t in best_ts[1:]:
        if round(t - cur[-1], 2) <= 0.011:
            cur.append(t)
        else:
            runs.append(cur)
            cur = [t]
    runs.append(cur)
    widest = max(runs, key=len)
    best_t = widest[len(widest) // 2]
    best = next(r for r in scored if r["threshold"] == best_t)
    m.update({"xv.best_threshold": best_t, "xv.plateau_low": widest[0], "xv.plateau_high": widest[-1],
              "xv.open_set_accuracy": best["accuracy"], "xv.wrong_match_rate": best["wrong_match_rate"],
              "xv.correct_match": best["correct_match"], "xv.missed_match": best["missed_match"],
              "xv.correct_reject": best["correct_reject"], "xv.wrong_match": best["wrong_match"]})
    if best.get("same_camera.accuracy") is not None:
        m["xv.same_camera.accuracy"] = best["same_camera.accuracy"]
    m["xv.unverified_matches"] = best.get("unverified_matches", 0)
    m["xv.cross_camera.unverified_matches"] = best.get("cross_camera.unverified_matches", 0)
    for t in cfg["xvideo"].get("report_thresholds", []):
        r = next((x for x in scored if abs(x["threshold"] - t) < 1e-9), None)
        if r:
            m[f"xv.accuracy@{t}"] = r["accuracy"]
            m[f"xv.wrong_match_rate@{t}"] = r["wrong_match_rate"]
    write_csv(os.path.join(run_dir, "classification", "xv_matches.csv"), xv_open_set(srcs, best_t))
    return m, missing


def eval_env(run_dir, cfg, gt):
    out = os.path.join(run_dir, "outputs")
    videos = runlog.read_json(os.path.join(out, "videos.json"), {})
    rows = []

    def check(name, venv, value, expected, passed):
        rows.append({"check": name, "venv": venv, "value": value, "expected": expected, "passed": bool(passed)})

    reports = {}
    for venv in cfg["env"]["venvs"]:
        rep = runlog.read_json(os.path.join(out, f"env_{venv}.json"), {})
        reports[venv] = rep
        check("funciona", venv, rep.get("error") or "ok", "ok", rep.get("ok"))
        for video, d in (rep.get("decode") or {}).items():
            v = videos.get(video, {})
            expected = round(v.get("fps", 0) * v.get("duration_s", 0))
            check(f"fotogramas {os.path.basename(video)}", venv, d.get("frames_decoded"),
                  f"{expected} (+-1)", d.get("opened") and abs(d.get("frames_decoded", 0) - expected) <= 1)
    track = reports.get("track", {}).get("probe", {})
    faces = reports.get("faces", {}).get("probe", {})
    check("GPU YOLO26m", "track", track.get("yolo_device"), "cuda", track.get("yolo_cuda"))
    check("GPU codificador ReID", "track", ",".join(track.get("onnx_providers", [])),
          "CUDAExecutionProvider primero", track.get("onnx_cuda"))
    check("GPU RetinaFace", "faces", ",".join(faces.get("tf_gpus", [])) or "sin GPU", "GPU:0", faces.get("tf_cuda"))
    outside = runlog.read_json(os.path.join(out, "outside.json"), {}).get("after", {})
    written = [f for s in outside.values() for f in s.get("written", [])]
    check("sin archivos fuera del repositorio", "", len(written), 0, not written)
    write_csv(os.path.join(run_dir, "classification", "env_checks.csv"), rows)
    passed = sum(r["passed"] for r in rows)
    m = {"env.checks": len(rows), "env.passed": passed, "env.pass_rate": passed / len(rows) if rows else None,
         "env.outside_files_written": len(written)}
    for venv, rep in reports.items():
        for k, v in (rep.get("packages") or {}).items():
            m[f"env.{venv}.{k}"] = v
    for k in ("gpu", "driver", "gpu_memory"):
        if reports.get("track", {}).get(k):
            m[f"env.{k}"] = reports["track"][k]
    return m, []


EVALUATORS = {"detect": eval_detect, "pipeline": eval_pipeline, "embed": eval_embed,
              "cluster": eval_cluster, "xvideo": eval_xvideo, "env": eval_env}


def kind_of(run_dir, meta):
    if meta.get("kind"):
        return meta["kind"]
    if os.path.exists(os.path.join(run_dir, "outputs", "detections.json")):
        return "detect"
    if os.path.exists(os.path.join(run_dir, "outputs", "candidates")):
        return "pipeline"
    return None


def evaluate_run(run_dir):
    meta = runlog.read_json(os.path.join(run_dir, "run.json"))
    cfg = runlog.read_json(os.path.join(run_dir, "config.json"))
    if not meta or not cfg or meta.get("status") != "ok":
        return None
    kind = kind_of(run_dir, meta)
    if kind is None:
        return None
    gt = load_gt(cfg["video"])
    try:
        m, missing = EVALUATORS[kind](run_dir, cfg, gt)
    except Exception as e:
        tb = traceback.format_exc()
        err = f"{type(e).__name__}: {e}"
        with open(os.path.join(run_dir, "events.jsonl"), "a") as f:
            f.write(json.dumps({"ts": runlog.utc_now(), "stage": "evaluate", "event": "error",
                                "message": err, "traceback": tb}) + "\n")
        m = {"_error": err, "_traceback": tb, "_missing": [f"falló la evaluación: {err}"],
             "_evaluated_utc": runlog.utc_now()}
        runlog.write_json(os.path.join(run_dir, "metrics.json"), m)
        print(f"  [invalid] {runlog.rel_exp(run_dir)}: {err}", flush=True)
        return m
    people_rows = m.pop("_people_rows", None)
    if people_rows is not None:
        write_csv(os.path.join(run_dir, "classification", "e2e_people.csv"), people_rows)
    for stage, t in meta.get("timing_s", {}).items():
        m[f"time.{stage}_s"] = t
    m["_missing"] = missing
    m["_evaluated_utc"] = runlog.utc_now()
    runlog.write_json(os.path.join(run_dir, "metrics.json"), m)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-id")
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()
    pattern = "*" if args.all else args.test_id
    run_dirs = sorted(d for d in glob.glob(os.path.join(runlog.RUNS_DIR, pattern, "*"))
                      if os.path.isdir(d) and not d.startswith(os.path.join(runlog.RUNS_DIR, "_cache")))
    n = 0
    for d in run_dirs:
        if evaluate_run(d) is not None:
            n += 1
    path = registry.rebuild()
    print(f"{n} ejecuciones evaluadas; registro: {runlog.rel_exp(path)}")
    from harness import export_excel
    xlsx, _ = export_excel.export()
    print(f"libro: {runlog.rel_exp(xlsx)}")


if __name__ == "__main__":
    main()
