import argparse
import os
import sys
import time

from harness import runlog


def decode(path):
    import cv2
    t0 = time.perf_counter()
    cap = cv2.VideoCapture(path)
    opened = cap.isOpened()
    n = 0
    while opened and cap.grab():
        n += 1
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return {"opened": opened, "frames_decoded": n, "fps": fps,
            "decode_s": round(time.perf_counter() - t0, 2)}


def first_frame(path):
    import cv2
    cap = cv2.VideoCapture(path)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"no se pudo leer el primer fotograma de {path}")
    return frame


def probe_track(video):
    import onnxruntime as ort
    import torch
    from ultralytics import YOLO
    out = {"torch_cuda": torch.cuda.is_available(),
           "torch_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    session = ort.InferenceSession(os.path.join(runlog.PIPELINE_DIR, "yolo26m-reid.onnx"),
                                   providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    out["onnx_providers"] = session.get_providers()
    out["onnx_cuda"] = session.get_providers()[0] == "CUDAExecutionProvider"
    model = YOLO(os.path.join(runlog.PIPELINE_DIR, "yolo26m.pt"))
    result = model.predict(first_frame(video), classes=[0], verbose=False)[0]
    out["yolo_device"] = str(result.boxes.data.device)
    out["yolo_cuda"] = out["yolo_device"].startswith("cuda")
    out["yolo_people_first_frame"] = len(result.boxes)
    return out


def probe_faces(video):
    sys.path.insert(0, runlog.PIPELINE_DIR)
    import face_recognition
    import tensorflow as tf
    from retinaface import RetinaFace
    gpus = tf.config.list_physical_devices("GPU")
    faces = RetinaFace.detect_faces(first_frame(video), threshold=face_recognition.FACE_CONFIDENCE_THRESHOLD)
    return {"tf_version": tf.__version__, "tf_gpus": [g.name for g in gpus], "tf_cuda": bool(gpus),
            "retinaface_faces_first_frame": len(faces) if isinstance(faces, dict) else 0}


def probe_eval(video):
    import numpy as np
    from sklearn.cluster import HDBSCAN
    from sklearn.metrics import pairwise_distances
    rng = np.random.default_rng(0)
    x = np.concatenate([np.array(c) + rng.normal(scale=0.01, size=(10, 2)) for c in ([0, 0], [0.2, 0], [10, 0])])
    labels = HDBSCAN(min_cluster_size=2, min_samples=1, cluster_selection_epsilon=0.5,
                     metric="precomputed", copy=True).fit_predict(pairwise_distances(x))
    merged = len(set(labels[:20])) == 1 and len(set(labels[20:])) == 1 and labels[0] != labels[20]
    if not merged:
        raise RuntimeError(f"HDBSCAN con cluster_selection_epsilon dio etiquetas inesperadas: {labels.tolist()}")
    return {"hdbscan_epsilon_ok": True, "hdbscan_clusters": int(labels.max() + 1)}


def probe_worker(video):
    import boto3
    return {"boto3": boto3.__version__}


PROBES = {"track": probe_track, "faces": probe_faces, "eval": probe_eval, "worker": probe_worker}
DECODES = ("track", "faces")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--venv", required=True, choices=sorted(PROBES))
    ap.add_argument("--out", required=True)
    ap.add_argument("--videos", nargs="+", required=True)
    args = ap.parse_args()
    report = {"venv": args.venv, "executable": sys.executable, **runlog.env_info()}
    try:
        report["probe"] = PROBES[args.venv](args.videos[0])
        report["ok"] = True
    except Exception as e:
        report["ok"] = False
        report["error"] = f"{type(e).__name__}: {e}"
    if args.venv in DECODES:
        report["decode"] = {runlog.rel_exp(v): decode(v) for v in args.videos}
    runlog.write_json(args.out, report)
    raise SystemExit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
