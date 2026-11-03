import argparse
import json
import os
import sys
import time

os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

import cv2

from harness import runlog

sys.path.insert(0, runlog.PIPELINE_DIR)
import face_recognition as fr


def instrument(events, manifest, faces_out):
    ctx = {}

    class Cv2Proxy:
        def __getattr__(self, name):
            return getattr(cv2, name)

        @staticmethod
        def imread(path, *a):
            ctx["path"] = path
            return cv2.imread(path, *a)
    fr.cv2 = Cv2Proxy()

    orig_best = fr.best_face

    def best_face(pixel_array):
        t0 = time.perf_counter()
        score, area = orig_best(pixel_array)
        ctx.update(score=score, area=area, infer_ms=round((time.perf_counter() - t0) * 1000, 1))
        return score, area
    fr.best_face = best_face

    orig_eval = fr.evaluate_candidate

    def evaluate_candidate(crop):
        cand = orig_eval(crop)
        track_id = os.path.basename(os.path.dirname(ctx["path"]))
        name = os.path.basename(ctx["path"])
        area, score = ctx.get("area"), ctx.get("score", 0.0)
        frame_idx = manifest.get("tracks", {}).get(track_id, {}).get(name)
        person_box = manifest.get("boxes", {}).get(track_id, {}).get(name)
        face_box_frame = None
        if area is not None and person_box is not None:
            face_box_frame = [area[0] + person_box[0], area[1] + person_box[1],
                              area[2] + person_box[0], area[3] + person_box[1]]
        if cand is None:
            if area is None:
                reason = "no_face"
            elif score < fr.FACE_CONFIDENCE_THRESHOLD:
                reason = "below_threshold"
            elif fr.MIN_FACE_PX and min(area[2] - area[0], area[3] - area[1]) < fr.MIN_FACE_PX:
                reason = "too_small"
            else:
                reason = "crop_failed"
        else:
            reason = None
            tight = fr.crop_face(crop, area, fr.FACE_MARGIN, fr.FACE_MARGIN, fr.FACE_MARGIN)
            out_dir = os.path.join(faces_out, "candidate_faces", track_id)
            os.makedirs(out_dir, exist_ok=True)
            cv2.imwrite(os.path.join(out_dir, name), tight)
        events.emit("candidate", track_id=track_id, crop=name, frame_idx=frame_idx,
                    score=round(score, 5), face_box_crop=list(area) if area else None,
                    face_box_frame=face_box_frame, person_box=person_box,
                    face_px=cand["face_px"] if cand else None,
                    sharpness=round(cand["sharpness"], 2) if cand else None,
                    rank_key=[round(float(v), 4) for v in fr.rank_key(cand)] if cand else None,
                    accepted=cand is not None, reject_reason=reason, infer_ms=ctx.get("infer_ms"))
        return cand
    fr.evaluate_candidate = evaluate_candidate

    orig_process = fr.process_track

    def process_track(track_id, folder, faces_dir):
        face = orig_process(track_id, folder, faces_dir)
        events.emit("track_result", track_id=track_id,
                    chosen_crop=face["candidate"] if face else None,
                    score=face["score"] if face else None,
                    n_candidates=len(os.listdir(folder)))
        return face
    fr.process_track = process_track


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--out", required=True, help="directorio de rostros")
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = json.load(f)
    for key, value in cfg["stage2"].items():
        if not hasattr(fr, key):
            raise SystemExit(f"face_recognition.py no tiene la constante {key}")
        setattr(fr, key, value)

    manifest = runlog.read_json(os.path.join(args.candidates, fr.MANIFEST_NAME), {})
    events = runlog.EventLog(os.path.join(args.run_dir, "events.jsonl"), "stage2")
    events.emit("stage_start", config_subset={"stage2": cfg["stage2"]})
    instrument(events, manifest, args.out)

    import tensorflow as tf
    t0 = time.perf_counter()
    sys.argv = ["face_recognition.py", "--candidates-dir", args.candidates, "--faces-dir", args.out]
    fr.main()
    elapsed = time.perf_counter() - t0
    results = runlog.read_json(os.path.join(args.out, fr.RESULTS_NAME), {"faces": []})
    summary = {"tracks_checked": results.get("tracks_checked"), "faces": len(results["faces"]),
               "device": "GPU" if tf.config.list_physical_devices("GPU") else "CPU",
               "elapsed_s": round(elapsed, 2), "peak_rss_mb": round(runlog.peak_rss_mb())}
    runlog.write_json(os.path.join(args.out, "stage2_summary.json"), summary)
    events.emit("stage_end", **summary)
    events.close()


if __name__ == "__main__":
    main()
