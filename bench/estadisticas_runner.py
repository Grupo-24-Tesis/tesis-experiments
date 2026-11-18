import argparse
import os
import time

from harness import runlog

YOLO_WEIGHTS = os.path.join(runlog.PIPELINE_DIR, "yolo26m.pt")
PERSON_CLASS = 0
CHECKS_PER_S = 2
CONF = 0.25
IMGSZ = 640


def settings():
    return {"weights": os.path.basename(YOLO_WEIGHTS), "classes": [PERSON_CLASS], "conf": CONF,
            "imgsz": IMGSZ, "checks_per_s": CHECKS_PER_S}


def check_step(fps):
    return max(1, round(fps / CHECKS_PER_S))


def read_checks(cap, step, person_heights):
    checks = []
    unreadable = 0
    frame_idx = -1
    while cap.grab():
        frame_idx += 1
        if frame_idx % step:
            continue
        ok, frame = cap.retrieve()
        if not ok:
            unreadable += 1
            continue
        checks.append([frame_idx, [round(float(h), 1) for h in person_heights(frame)]])
    return checks, unreadable, frame_idx + 1


def load_detector():
    from ultralytics import YOLO
    model = YOLO(YOLO_WEIGHTS)

    def person_heights(frame):
        result = model.predict(source=frame, classes=[PERSON_CLASS], conf=CONF, imgsz=IMGSZ,
                               verbose=False)[0]
        if result.boxes is None or len(result.boxes) == 0:
            return []
        return [float(box[3] - box[1]) for box in result.boxes.xyxy.cpu().numpy()]
    return person_heights


def parse_args():
    parser = argparse.ArgumentParser(description="Revisa un video con YOLO dos veces por segundo, "
                                                 "como la compuerta de presencia, y guarda cuántas "
                                                 "personas hay en cada revisión y la altura de sus "
                                                 "cajas")
    parser.add_argument("--video", required=True, help="video de entrada")
    parser.add_argument("--out", required=True, help="archivo JSON de salida")
    return parser.parse_args()


def main():
    args = parse_args()
    import cv2
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"No se pudo abrir el video: {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    step = check_step(fps)
    started = time.perf_counter()
    person_heights = load_detector()
    checks, unreadable, frames = read_checks(cap, step, person_heights)
    cap.release()
    elapsed = time.perf_counter() - started
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    runlog.write_json(args.out, {"video": os.path.basename(args.video), "fps": fps, "width": width,
                                 "height": height, "frames": frames, "check_step": step,
                                 "settings": settings(), "unreadable": unreadable,
                                 "elapsed_s": round(elapsed, 2), "checks": checks})
    print(f"{len(checks)} revisiones de {frames} fotogramas en {elapsed:.1f} s "
          f"({unreadable} ilegibles)")


if __name__ == "__main__":
    main()
