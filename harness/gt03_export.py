import argparse
import glob
import json
import os

import cv2

from harness import items as items_mod
from harness import runlog
from harness.metrics.detection import iou

SUGGEST_MAX_FRAMES = 8
SUGGEST_IOU = 0.3
THUMB_MARGIN = 0.8


def suggest(gt_frames, frame_idx, box):
    best, best_score = None, SUGGEST_IOU
    for k, fr in gt_frames.items():
        gap = abs(int(k) - frame_idx)
        if not fr.get("reviewed") or gap > SUGGEST_MAX_FRAMES:
            continue
        for f in fr["faces"]:
            if f.get("identity") and not f.get("ignore"):
                o = iou(f["box"], box)
                if o > best_score:
                    best, best_score = f["identity"], o
    return best


def export_run(run_dir, gt_dir, doc, gt_frames):
    events = os.path.join(run_dir, "events.jsonl")
    cands = os.path.realpath(os.path.join(run_dir, "outputs", "candidates"))
    added = 0
    with open(events) as f:
        for line in f:
            e = json.loads(line)
            if e["event"] != "candidate" or not e["accepted"] or e["face_box_frame"] is None:
                continue
            fi, box = e["frame_idx"], e["face_box_frame"]
            if any(it["frame_idx"] == fi and iou(it["box"], box) >= items_mod.LABEL_IOU
                   for it in doc["items"].values()):
                continue
            key = f"{fi:06d}_{'_'.join(str(int(v)) for v in box)}"
            crop = cv2.imread(os.path.join(cands, e["track_id"], e["crop"]))
            if crop is None:
                continue
            x1, y1, x2, y2 = e["face_box_crop"]
            h, w = crop.shape[:2]
            mx, my = (x2 - x1) * THUMB_MARGIN, (y2 - y1) * THUMB_MARGIN
            thumb = crop[int(max(0, y1 - my)):int(min(h, y2 + 2 * my)), int(max(0, x1 - mx)):int(min(w, x2 + mx))]
            cv2.imwrite(os.path.join(gt_dir, "crops", key + ".jpg"), thumb)
            doc["items"][key] = {"frame_idx": fi, "box": box, "label": None,
                                 "suggested": suggest(gt_frames, fi, box),
                                 "source": f"{os.path.basename(os.path.dirname(run_dir))} track {e['track_id']}",
                                 "added_utc": runlog.utc_now()}
            added += 1
    return added


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-id")
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()
    pattern = "*" if args.all else args.test_id
    total = 0
    for run_dir in sorted(glob.glob(os.path.join(runlog.RUNS_DIR, pattern, "*"))):
        meta = runlog.read_json(os.path.join(run_dir, "run.json"), {})
        if meta.get("kind") != "pipeline" or meta.get("status") != "ok":
            continue
        cfg = runlog.read_json(os.path.join(run_dir, "config.json"))
        gt_dir = items_mod.gt_dir(cfg)
        os.makedirs(os.path.join(gt_dir, "crops"), exist_ok=True)
        path = os.path.join(gt_dir, "crop_labels.json")
        doc = runlog.read_json(path, {"items": {}})
        frames = (runlog.read_json(os.path.join(gt_dir, "frames.json")) or {"frames": {}})["frames"]
        n = export_run(run_dir, gt_dir, doc, frames)
        runlog.write_json(path, doc)
        total += n
        print(f"{runlog.rel_exp(run_dir)}: +{n} rostros por etiquetar")
    print(f"{total} rostros nuevos en cola para GT-03")


if __name__ == "__main__":
    main()
