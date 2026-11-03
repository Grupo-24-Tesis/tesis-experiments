import argparse
import json
import time
import traceback

from harness import runlog


def probe_track():
    import torch
    import onnxruntime as ort
    import ultralytics
    return {"torch_cuda": torch.cuda.is_available(),
            "torch_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "onnx_providers": ort.get_available_providers(),
            "ultralytics": ultralytics.__version__}


def probe_faces():
    import tensorflow as tf
    import retinaface
    import deepface
    import mtcnn
    return {"tf_gpus": [d.name for d in tf.config.list_physical_devices("GPU")],
            "tf_version": tf.__version__}


def probe_eval():
    import sklearn
    from sklearn.cluster import HDBSCAN
    return {"sklearn": sklearn.__version__, "hdbscan_builtin": True}


def decode_check(video):
    import cv2
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        return {"opened": False}
    fps = cap.get(cv2.CAP_PROP_FPS)
    reported = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    t0, n, shape = time.perf_counter(), 0, None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        shape = shape or list(frame.shape)
        n += 1
    cap.release()
    return {"opened": True, "fps": fps, "frames_reported": reported, "frames_decoded": n,
            "frame_shape": shape, "decode_s": round(time.perf_counter() - t0, 2)}


PROBES = {"track": probe_track, "faces": probe_faces, "eval": probe_eval}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--venv", required=True, choices=PROBES)
    ap.add_argument("--video")
    args = ap.parse_args()
    report = {"venv": args.venv, **runlog.env_info()}
    try:
        report["probe"] = PROBES[args.venv]()
        report["ok"] = True
    except Exception:
        report["ok"] = False
        report["error"] = traceback.format_exc()
    if args.video and args.venv in ("track", "faces"):
        report["decode"] = decode_check(runlog.exp_path(args.video))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
