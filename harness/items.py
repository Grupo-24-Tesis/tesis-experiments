import json
import os

from harness import runlog
from harness.metrics.detection import in_scope, iou, face_size

GT_ROOT = runlog.GT_ROOT
LABEL_IOU = 0.5


def gt_dir(cfg):
    return os.path.join(GT_ROOT, os.path.splitext(os.path.basename(cfg["video"]))[0])


def gt_items(cfg):
    d = gt_dir(cfg)
    doc = runlog.read_json(os.path.join(d, "frames.json"))
    if not doc:
        return []
    poses = set(cfg["items"]["poses"])
    out = []
    for key in sorted(doc["frames"], key=int):
        fr = doc["frames"][key]
        if not fr.get("reviewed"):
            continue
        for i, f in enumerate(fr["faces"]):
            if f.get("identity") and in_scope(f, cfg["gt"]["min_face_px"]) and f.get("pose") in poses:
                out.append({"item_id": f"gt_{int(key):06d}_{i}", "image": runlog.rel_exp(os.path.join(d, "frames", f"{int(key):06d}.jpg")),
                            "box": f["box"], "identity": f["identity"], "time_s": fr["time_s"],
                            "frame_idx": int(key), "pose": f["pose"], "occlusion": f["occlusion"],
                            "size_px": round(face_size(f["box"]), 1), "track_id": None})
    return out


def crop_label(labels, frame_idx, box):
    best, best_iou = None, LABEL_IOU
    for item in labels.values():
        if item["frame_idx"] == frame_idx and item.get("label") and box is not None:
            o = iou(item["box"], box)
            if o >= best_iou:
                best, best_iou = item["label"], o
    return best


def pipeline_faces(run_dir, top_k):
    by_track = {}
    with open(os.path.join(run_dir, "events.jsonl")) as f:
        for line in f:
            e = json.loads(line)
            if e["event"] == "candidate" and e["accepted"]:
                by_track.setdefault(e["track_id"], []).append(e)
    out = []
    for track_id, cands in sorted(by_track.items(), key=lambda kv: int(kv[0])):
        cands.sort(key=lambda e: e["rank_key"], reverse=True)
        out += cands[:top_k]
    return out


def pipeline_items(cfg):
    run_dir = runlog.exp_path(cfg["items"]["pipeline_run"])
    labels = runlog.read_json(os.path.join(gt_dir(cfg), "crop_labels.json"), {"items": {}})["items"]
    faces_dir = os.path.realpath(os.path.join(run_dir, "outputs", "faces"))
    meta = runlog.read_json(os.path.join(run_dir, "run.json"), {})
    fps = meta.get("video", {}).get("fps") or 15.0
    out = []
    for e in pipeline_faces(run_dir, cfg["select"]["top_k"]):
        image = runlog.rel_exp(os.path.join(faces_dir, "candidate_faces", e["track_id"], e["crop"]))
        out.append({"item_id": f"trk{e['track_id']}_{e['crop'].rsplit('.', 1)[0]}", "image": image,
                    "box": None, "identity": crop_label(labels, e["frame_idx"], e["face_box_frame"]),
                    "time_s": e["frame_idx"] / fps if e["frame_idx"] is not None else None,
                    "frame_idx": e["frame_idx"], "track_id": e["track_id"],
                    "face_box_frame": e["face_box_frame"], "score": e["score"]})
    return out


CONTROL_EVERY = 10


def auto_items(cfg):
    d = gt_dir(cfg)
    doc = runlog.read_json(os.path.join(d, "frames_gt01.json"))
    if not doc:
        return []
    keys = sorted(doc["frames"], key=int)
    faces, sizes = [], []
    for key in keys:
        fr = doc["frames"][key]
        for i, f in enumerate(fr["faces"]):
            if f.get("ignore"):
                continue
            sizes.append(face_size(f["box"]))
            faces.append({"item_id": f"auto_{int(key):06d}_{i}", "image": runlog.rel_exp(os.path.join(d, "frames", f"{int(key):06d}.jpg")),
                          "box": f["box"], "identity": None, "time_s": fr["time_s"], "frame_idx": int(key),
                          "control": False, "size_px": round(face_size(f["box"]), 1), "track_id": None})
    if not sizes:
        return faces
    s = sorted(sizes)[len(sizes) // 2]
    w, h = doc["video"]["width"], doc["video"]["height"]
    controls = []
    for n, key in enumerate(keys[::CONTROL_EVERY]):
        boxes = [f["box"] for f in doc["frames"][key]["faces"]]
        for gy in range(4):
            box = None
            for gx in range(8):
                x1, y1 = (gx * w / 8 + n * 7) % (w - s), (gy * h / 4 + n * 5) % (h - s)
                cand = [round(x1, 1), round(y1, 1), round(x1 + s, 1), round(y1 + s, 1)]
                if all(iou(cand, b) == 0 for b in boxes):
                    box = cand
                    break
            if box:
                controls.append({"item_id": f"control_{int(key):06d}", "image": runlog.rel_exp(os.path.join(d, "frames", f"{int(key):06d}.jpg")),
                                 "box": box, "identity": None, "time_s": doc["frames"][key]["time_s"],
                                 "frame_idx": int(key), "control": True, "size_px": s, "track_id": None})
                break
    return sorted(faces + controls, key=lambda it: it["frame_idx"])


def build(cfg):
    source = cfg["items"]["source"]
    builders = {"gt": gt_items, "pipeline": pipeline_items, "auto": auto_items}
    return builders[source](cfg), source
