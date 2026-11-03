import argparse
import os
import time

os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

import cv2
import numpy as np

from harness import runlog

ALIGN_MARGIN = 0.5


def cut(image, box, margin):
    h, w = image.shape[:2]
    x1, y1, x2, y2 = box
    mx, my = (x2 - x1) * margin, (y2 - y1) * margin
    x1, y1 = int(max(0, x1 - mx)), int(max(0, y1 - my))
    x2, y2 = int(min(w, x2 + mx)), int(min(h, y2 + my))
    return image[y1:y2, x1:x2], (x1, y1)


def nearest_face(results, center):
    def dist(r):
        a = r["facial_area"]
        return (a["x"] + a["w"] / 2 - center[0]) ** 2 + (a["y"] + a["h"] / 2 - center[1]) ** 2
    return min(results, key=dist)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="ArcFace")
    ap.add_argument("--align", type=lambda s: s.lower() == "true", default=True)
    ap.add_argument("--normalization", default="base")
    args = ap.parse_args()

    from deepface import DeepFace
    items = runlog.read_json(args.items)
    os.makedirs(args.out, exist_ok=True)
    events = runlog.EventLog(os.path.join(args.out, "events.jsonl"), "embed")
    events.emit("stage_start", model=args.model, align=args.align, normalization=args.normalization,
                n_items=len(items))
    images, vectors, index = {}, [], []
    t_start = time.perf_counter()
    for it in items:
        if it["image"] not in images:
            images = {it["image"]: cv2.imread(runlog.exp_path(it["image"]))}
        img = images[it["image"]]
        t0 = time.perf_counter()
        aligned, ok, err = False, True, None
        try:
            if it.get("box"):
                face, _ = cut(img, it["box"], ALIGN_MARGIN if args.align else 0.0)
            else:
                face = img
            if args.align:
                res = DeepFace.represent(face, model_name=args.model, detector_backend="retinaface",
                                         align=True, enforce_detection=False,
                                         normalization=args.normalization)
                h, w = face.shape[:2]
                best = nearest_face(res, (w / 2, h / 2))
                aligned = best.get("face_confidence", 0) > 0
                vec = best["embedding"]
            else:
                res = DeepFace.represent(face, model_name=args.model, detector_backend="skip",
                                         align=False, enforce_detection=False,
                                         normalization=args.normalization)
                vec = res[0]["embedding"]
        except Exception as e:
            ok, err, vec = False, repr(e), None
        ms = round((time.perf_counter() - t0) * 1000, 1)
        vectors.append(vec)
        index.append({"item_id": it["item_id"], "identity": it.get("identity"), "time_s": it.get("time_s"),
                      "ok": ok, "aligned": aligned, "error": err})
        events.emit("embed", item_id=it["item_id"], model=args.model, ok=ok, aligned=aligned,
                    infer_ms=ms, error=err)
    dim = next((len(v) for v in vectors if v is not None), 0)
    mat = np.array([v if v is not None else [np.nan] * dim for v in vectors], dtype=np.float32)
    np.save(os.path.join(args.out, "embeddings.npy"), mat)
    runlog.write_json(os.path.join(args.out, "embeddings_index.json"), index)
    events.emit("stage_end", elapsed_s=round(time.perf_counter() - t_start, 2), dim=dim,
                n_ok=sum(i["ok"] for i in index), n_aligned=sum(i["aligned"] for i in index),
                peak_rss_mb=round(runlog.peak_rss_mb()))
    events.close()


if __name__ == "__main__":
    main()
