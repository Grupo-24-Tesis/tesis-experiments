import argparse
import os
import time

os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

import cv2

from harness import runlog
from harness.metrics.detection import iou

PRE_ANNOTATE_THRESHOLD = 0.3
MERGE_IOU = 0.5


def sample_frames(video, every_s, frame_ids=None):
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    wanted = set(frame_ids) if frame_ids is not None else None
    step = max(1, round(fps * every_s)) if wanted is None else None
    idx = -1
    try:
        while cap.grab():
            idx += 1
            if (idx in wanted) if wanted is not None else (idx % step == 0):
                ok, frame = cap.retrieve()
                if ok:
                    yield idx, frame
    finally:
        cap.release()


class RetinaFaceDetector:
    name = "retinaface"

    def __init__(self, threshold, allow_upscaling=True, **_):
        from retinaface import RetinaFace
        self._rf = RetinaFace
        self.threshold = threshold
        self.allow_upscaling = allow_upscaling

    def __call__(self, bgr):
        resp = self._rf.detect_faces(bgr, threshold=self.threshold,
                                     allow_upscaling=self.allow_upscaling)
        if not isinstance(resp, dict):
            return []
        return [{"box": [float(v) for v in f["facial_area"]], "score": float(f["score"]),
                 "landmarks": {k: [float(x) for x in v] for k, v in f.get("landmarks", {}).items()}}
                for f in resp.values()]


class MTCNNDetector:
    name = "mtcnn"

    def __init__(self, threshold, min_face_size=20, **_):
        from mtcnn import MTCNN
        self._m = MTCNN(device="CPU:0")
        self.threshold = threshold
        self.min_face_size = min_face_size

    def __call__(self, bgr):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        faces = self._m.detect_faces(rgb, box_format="xyxy", min_face_size=self.min_face_size,
                                     threshold_onet=self.threshold)
        return [{"box": [float(v) for v in f["box"]], "score": float(f["confidence"]),
                 "landmarks": {k: [float(x) for x in v] for k, v in f.get("keypoints", {}).items()}}
                for f in faces]


class PipelineDetector:
    name = "pipeline"

    def __init__(self, threshold, stage2=None, person_boxes=None, **_):
        import sys
        sys.path.insert(0, runlog.PIPELINE_DIR)
        import face_recognition as fr
        for key, value in (stage2 or {}).items():
            setattr(fr, key, value)
        fr.FACE_CONFIDENCE_THRESHOLD = threshold
        self.fr = fr
        self.person_boxes = person_boxes or {}
        self.frame_idx = None

    def __call__(self, bgr):
        h, w = bgr.shape[:2]
        found = []
        for x1, y1, x2, y2, _ in self.person_boxes.get(str(self.frame_idx), []):
            x1, y1, x2, y2 = int(max(0, x1)), int(max(0, y1)), int(min(w, x2)), int(min(h, y2))
            if x2 <= x1 or y2 <= y1:
                continue
            score, area = self.fr.best_face(bgr[y1:y2, x1:x2])
            if area is None:
                continue
            fx1, fy1, fx2, fy2 = area
            if min(fx2 - fx1, fy2 - fy1) < self.fr.MIN_FACE_PX:
                continue
            found.append({"box": [fx1 + x1, fy1 + y1, fx2 + x1, fy2 + y1], "score": float(score),
                          "landmarks": {}})
        merged = []
        for d in sorted(found, key=lambda d: d["score"], reverse=True):
            if all(iou(d["box"], m["box"]) < MERGE_IOU for m in merged):
                merged.append(d)
        return merged


DETECTORS = {"retinaface": RetinaFaceDetector, "mtcnn": MTCNNDetector, "pipeline": PipelineDetector}


def detect(detector, frame, upscale=1.0):
    if upscale != 1.0:
        frame = cv2.resize(frame, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_CUBIC)
    dets = detector(frame)
    if upscale != 1.0:
        for d in dets:
            d["box"] = [v / upscale for v in d["box"]]
            d["landmarks"] = {k: [v / upscale for v in p] for k, p in d["landmarks"].items()}
    return dets


def merge(dets_by_source):
    pool = sorted(((d, src) for src, ds in dets_by_source.items() for d in ds),
                  key=lambda x: x[0]["score"], reverse=True)
    merged = []
    for d, src in pool:
        for m in merged:
            if iou(m["box"], d["box"]) >= MERGE_IOU:
                m["sources"][src] = round(d["score"], 4)
                break
        else:
            merged.append({"box": [round(v, 1) for v in d["box"]], "score": d["score"],
                           "sources": {src: round(d["score"], 4)}})
    return merged


def pre_annotate(args):
    gt_dir = args.out
    frames_dir = os.path.join(gt_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    path = os.path.join(gt_dir, "frames.json")
    existing = runlog.read_json(path)
    if existing and any(f.get("reviewed") for f in existing["frames"].values()):
        raise SystemExit(f"{path} ya tiene fotogramas revisados; no se sobrescriben las etiquetas")
    dets = {name: DETECTORS[name](PRE_ANNOTATE_THRESHOLD) for name in ("retinaface", "mtcnn")}
    args.video = runlog.exp_path(args.video)
    video = runlog.video_info(args.video)
    frames = {}
    for frame_idx, frame in sample_frames(args.video, args.every_s):
        cv2.imwrite(os.path.join(frames_dir, f"{frame_idx:06d}.jpg"), frame,
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
        boxes = merge({name: det(frame) for name, det in dets.items()})
        frames[str(frame_idx)] = {
            "time_s": round(frame_idx / video["fps"], 3),
            "reviewed": False,
            "faces": [{"box": b["box"], "identity": None, "pose": None, "occlusion": None,
                       "ignore": False, "source": "auto", "auto_sources": b["sources"]}
                      for b in boxes],
        }
        print(f"fotograma {frame_idx}: {len(boxes)} cajas automáticas", flush=True)
    runlog.write_json(path, {"video": video, "sample_every_s": args.every_s,
                             "pre_annotate_threshold": PRE_ANNOTATE_THRESHOLD,
                             "created_utc": runlog.utc_now(), "frames": frames})
    print(f"Se escribieron {len(frames)} fotogramas en {path}")


def run_detector(args):
    gt = runlog.read_json(args.gt)
    frame_ids = sorted(int(k) for k in gt["frames"])
    cfg = runlog.read_json(args.config) if args.config else {}
    det = DETECTORS[args.detector](args.threshold, allow_upscaling=args.allow_upscaling,
                                   min_face_size=args.mtcnn_min_face, stage2=cfg.get("stage2"),
                                   person_boxes=runlog.read_json(args.person_boxes) if args.person_boxes else None)
    events = runlog.EventLog(os.path.join(args.out, "events.jsonl"), "detect")
    events.emit("stage_start", detector=args.detector, threshold=args.threshold,
                allow_upscaling=args.allow_upscaling, upscale=args.upscale,
                mtcnn_min_face=args.mtcnn_min_face)
    t0 = time.perf_counter()
    out = {}
    for frame_idx, frame in sample_frames(runlog.exp_path(args.video), None, frame_ids):
        t = time.perf_counter()
        det.frame_idx = frame_idx
        dets = detect(det, frame, args.upscale)
        ms = round((time.perf_counter() - t) * 1000, 1)
        out[str(frame_idx)] = dets
        events.emit("det_frame", frame_idx=frame_idx, detector=args.detector,
                    n_dets=len(dets), infer_ms=ms)
    events.emit("stage_end", elapsed_s=round(time.perf_counter() - t0, 2),
                peak_rss_mb=round(runlog.peak_rss_mb()))
    events.close()
    os.makedirs(os.path.join(args.out, "outputs"), exist_ok=True)
    runlog.write_json(os.path.join(args.out, "outputs", "detections.json"), out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True, help="directorio GT (pre-anotación) o directorio de la ejecución (detector)")
    ap.add_argument("--pre-annotate", action="store_true")
    ap.add_argument("--every-s", type=float, default=1.0)
    ap.add_argument("--gt", help="frames.json cuyos fotogramas se evalúan")
    ap.add_argument("--detector", choices=DETECTORS)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--allow-upscaling", type=lambda s: s.lower() == "true", default=True)
    ap.add_argument("--upscale", type=float, default=1.0)
    ap.add_argument("--mtcnn-min-face", type=int, default=20)
    ap.add_argument("--config", help="configuración de la ejecución (constantes de stage2 para --detector pipeline)")
    ap.add_argument("--person-boxes", help="person_boxes.json para --detector pipeline")
    args = ap.parse_args()
    if args.pre_annotate:
        pre_annotate(args)
    else:
        run_detector(args)


if __name__ == "__main__":
    main()
