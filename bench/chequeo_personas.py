import argparse
import json
import os

import cv2

PERSON_CLASS = 0


def parse_args():
    parser = argparse.ArgumentParser(description="Revisa un video con YOLO cada cierto numero de "
                                                 "fotogramas y guarda si hay alguna persona")
    parser.add_argument("--video", required=True, help="video de entrada")
    parser.add_argument("--out", required=True, help="archivo JSON de salida")
    parser.add_argument("--step", type=int, default=5, help="revisar uno de cada N fotogramas")
    parser.add_argument("--conf", type=float, default=0.05, help="confianza minima de YOLO")
    parser.add_argument("--imgsz", type=int, default=640, help="tamano de imagen de YOLO")
    parser.add_argument("--weights", required=True, help="pesos de YOLO")
    return parser.parse_args()


def main():
    args = parse_args()
    from ultralytics import YOLO
    model = YOLO(args.weights)
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"No se pudo abrir el video: {args.video}")
    checks = []
    index = -1
    while cap.grab():
        index += 1
        if index % args.step:
            continue
        ok, frame = cap.retrieve()
        if not ok:
            checks.append([index, True])
            continue
        result = model.predict(source=frame, classes=[PERSON_CLASS], conf=args.conf,
                               imgsz=args.imgsz, verbose=False)[0]
        checks.append([index, len(result.boxes) > 0])
    cap.release()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"step": args.step, "frames": index + 1, "checks": checks}, f)
    print(f"{len(checks)} revisiones, {sum(1 for _, p in checks if p)} con persona")


if __name__ == "__main__":
    main()
