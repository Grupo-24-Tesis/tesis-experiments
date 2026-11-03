import argparse
import json

import cv2
from ultralytics import YOLO

from harness import runlog


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = json.load(f)
    s1 = cfg["stage1"]
    wanted = {int(k) for k in runlog.read_json(args.gt)["frames"]}
    model = YOLO(s1["YOLO_WEIGHTS"])
    cap = cv2.VideoCapture(runlog.exp_path(cfg["video"]))
    out, idx = {}, -1
    while cap.grab():
        idx += 1
        if idx not in wanted:
            continue
        ok, frame = cap.retrieve()
        if not ok:
            continue
        r = model.predict(frame, classes=[0], conf=s1["YOLO_CONF"], iou=s1["YOLO_IOU"],
                          imgsz=s1["YOLO_IMGSZ"], verbose=False)[0]
        boxes = r.boxes.xyxy.cpu().numpy().tolist() if r.boxes is not None else []
        scores = r.boxes.conf.cpu().numpy().tolist() if r.boxes is not None else []
        out[str(idx)] = [[*map(float, b), float(s)] for b, s in zip(boxes, scores)]
    cap.release()
    runlog.write_json(args.out, out)
    print(f"cajas de persona para {len(out)} fotogramas")


if __name__ == "__main__":
    main()
