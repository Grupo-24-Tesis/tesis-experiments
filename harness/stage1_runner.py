import argparse
import json
import os
import sys
import time

import cv2
import yaml

from harness import runlog

sys.path.insert(0, runlog.PIPELINE_DIR)
import tracking


def write_tracker_yaml(path, tracker_cfg):
    with open(path, "w") as f:
        yaml.safe_dump(tracker_cfg, f, sort_keys=False)


def instrument(model, events, detect_every, scene_cut_threshold):
    state = {"calls": 0}
    orig_track = model.track

    def track(*args, **kwargs):
        t0 = time.perf_counter()
        results = orig_track(*args, **kwargs)
        r = results[0]
        n_dets = 0 if r.boxes is None else len(r.boxes)
        ids = [] if r.boxes is None or r.boxes.id is None else [int(i) for i in r.boxes.id.cpu().numpy()]
        events.emit("frame", frame_idx=state["calls"] * detect_every, n_dets=n_dets,
                    n_tracked=len(ids), track_ids=ids,
                    infer_ms=round((time.perf_counter() - t0) * 1000, 1))
        state["calls"] += 1
        return results
    model.track = track

    real_compare = cv2.compareHist

    class Cv2Proxy:
        def __getattr__(self, name):
            return getattr(cv2, name)

        @staticmethod
        def compareHist(a, b, method):
            corr = real_compare(a, b, method)
            if corr < scene_cut_threshold:
                events.emit("scene_cut", frame_idx=state["calls"] * detect_every,
                            correlation=round(corr, 4))
            return corr
    tracking.cv2 = Cv2Proxy()

    orig_thin = tracking.thin_candidates

    def thin(track_id, out_dir):
        before = set(tracking.candidate_frames[track_id])
        orig_thin(track_id, out_dir)
        events.emit("crop_thinned", track_id=track_id,
                    removed=sorted(before - set(tracking.candidate_frames[track_id])),
                    sample_every=tracking.sample_every[track_id])
    tracking.thin_candidates = thin


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out", required=True, help="directorio de candidatos")
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = json.load(f)

    for key, value in cfg["stage1"].items():
        if not hasattr(tracking, key):
            raise SystemExit(f"tracking.py no tiene la constante {key}")
        setattr(tracking, key, value)
    tracker_yaml = os.path.join(args.run_dir, "tracker.yaml")
    write_tracker_yaml(tracker_yaml, cfg["tracker"])
    tracking.TRACKER = tracker_yaml

    from ultralytics import YOLO
    events = runlog.EventLog(os.path.join(args.run_dir, "events.jsonl"), "stage1")
    events.emit("stage_start", config_subset={"stage1": cfg["stage1"], "tracker": cfg["tracker"]})
    video = runlog.exp_path(cfg["video"])
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    detect_every = max(1, round(fps * tracking.DETECT_INTERVAL_S))

    t0 = time.perf_counter()
    model = YOLO(tracking.YOLO_WEIGHTS)
    instrument(model, events, detect_every, tracking.SCENE_CUT_CORRELATION)
    fps, detect_every = tracking.people_tracking(model, video, args.out)
    tracking.write_manifest(args.out, video, fps, detect_every)
    elapsed = time.perf_counter() - t0

    for track_id, frames in sorted(tracking.candidate_frames.items()):
        boxes = tracking.candidate_boxes.get(track_id, {})
        for crop, frame_idx in sorted(frames.items()):
            events.emit("crop_saved", track_id=track_id, crop=crop, frame_idx=frame_idx,
                        box=boxes.get(crop))
    summary = {
        "n_tracks": len(tracking.candidate_frames),
        "n_crops": sum(len(v) for v in tracking.candidate_frames.values()),
        "fps": fps, "detect_every": detect_every,
        "device": str(model.device),
        "elapsed_s": round(elapsed, 2),
        "peak_rss_mb": round(runlog.peak_rss_mb()),
    }
    runlog.write_json(os.path.join(args.out, "stage1_summary.json"), summary)
    events.emit("stage_end", **summary)
    events.close()
    print(f"stage1: {summary['n_tracks']} tracks, {summary['n_crops']} recortes en {elapsed:.1f}s")


if __name__ == "__main__":
    main()
