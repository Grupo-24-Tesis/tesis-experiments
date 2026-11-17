import argparse
import json
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))

from bench import metrics

DEFAULT_PIPELINE_DIR = "/data/TesisFaceRecognition/wt/tesis_face_recognition_pipeline"
DEFAULT_CHOKEPOINT_DIR = "/data/TesisFaceRecognition/chokepoint"
FPS = 30
WIDTH = 800
HEIGHT = 600
LABELLED_SOURCES = ["P2E_S1_C1.1", "P2E_S1_C1.2"]
BUSY_SOURCE = "P2E_S5_C1"
CHECK_STEP = 5
CHECK_CONF = 0.05
CHECK_IMGSZ = 640
EMPTY_MARGIN_S = 1.0
EMPTY_MIN_S = 2.0
MERGE_GAP = 3
UNSCORED_ENTRY = [metrics.UNSCORED_SOURCE, -1]
EMPTY_NAME = "vacios.json"
CHECKS_SUFFIX = ".checks.json"
LARGE_WEIGHTS = [1.0, 0.7, 1.0, 0.9, 0.5, 0.8]
LARGE_STATIC_INDEX = 2
LARGE_STATIC_S = 300
LARGE_MIN_S = 120
LARGE_MAX_S = 600
BUSY_CLIP_S = 20
AFTER_FRAMES = 20


def chokepoint_dir():
    return os.environ.get("TESIS_CHOKEPOINT_DIR", DEFAULT_CHOKEPOINT_DIR)


def pipeline_dir():
    return os.environ.get("TESIS_PIPELINE_DIR", DEFAULT_PIPELINE_DIR)


def default_output_dir():
    return os.path.join(chokepoint_dir(), "runs", "gating", "videos_prueba")


def source_labels(gt, frame_map, frames):
    labels = []
    for index in range(frames):
        people = metrics.people_at(gt, frame_map, index)
        labels.append(None if people is None else [pid for pid, _ in people])
    return labels


def presence_runs(labels, merge_gap):
    runs = []
    for index, label in enumerate(labels):
        if not label:
            continue
        if runs and index - runs[-1][1] - 1 <= merge_gap:
            runs[-1] = (runs[-1][0], index)
        else:
            runs.append((index, index))
    return runs


def empty_stretches(labels, checks, step, margin, min_len):
    total = len(labels)
    blocked = [label is None for label in labels]
    for index, label in enumerate(labels):
        if label:
            for near in range(max(0, index - margin), min(total, index + margin + 1)):
                blocked[near] = True
    for frame, person in checks:
        if person:
            for near in range(max(0, frame - step + 1), min(total, frame + step)):
                blocked[near] = True
    stretches = []
    start = None
    for index in range(total + 1):
        if index < total and not blocked[index]:
            if start is None:
                start = index
            continue
        if start is not None and index - start >= min_len:
            stretches.append((start, index - 1))
        start = None
    return stretches


def pingpong_indices(start, end, count):
    length = end - start + 1
    if length <= 1:
        return [start] * count
    period = 2 * (length - 1)
    indices = []
    for step in range(count):
        position = step % period
        indices.append(start + (position if position < length else period - position))
    return indices


def first_appearances(labels):
    first = {}
    for index, label in enumerate(labels):
        for pid in label or []:
            first.setdefault(pid, index)
    return first


def densest_arrivals(first_frames, window):
    best_start, best_count = 0, 0
    for start in sorted(set(first_frames.values())):
        count = sum(1 for frame in first_frames.values() if start <= frame < start + window)
        if count > best_count:
            best_start, best_count = start, count
    return best_start, best_count


def distribute_empty(total, gaps, static_index, static_frames, weights, min_frames, max_frames):
    if len(weights) != gaps:
        raise ValueError("la cantidad de pesos debe ser igual a la cantidad de tramos vacíos")
    others = [i for i in range(gaps) if i != static_index]
    remaining = total - static_frames
    weight_sum = sum(weights[i] for i in others)
    frames = [0] * gaps
    frames[static_index] = static_frames
    assigned = 0
    for i in others[:-1]:
        frames[i] = int(remaining * weights[i] / weight_sum)
        assigned += frames[i]
    frames[others[-1]] = remaining - assigned
    for i in others:
        if not min_frames <= frames[i] <= max_frames:
            raise ValueError(f"el tramo vacío {i} queda con {frames[i]} fotogramas, fuera de "
                             f"[{min_frames}, {max_frames}]")
    return frames


def expand_segment(segment):
    source = segment["source"]
    if segment["kind"] == "empty":
        indices = pingpong_indices(segment["start"], segment["end"], segment["frames"])
    elif segment["kind"] == "clip":
        indices = list(range(segment["start"], segment["start"] + segment["frames"]))
    else:
        indices = [segment["start"]] * segment["frames"]
    return [(source, index) for index in indices]


def segment_map_entries(segment, sources):
    if segment["kind"] == "empty":
        return [None] * segment["frames"]
    frame_map = sources[segment["source"]]["map"]
    entries = []
    for source, index in expand_segment(segment):
        if frame_map is None or index >= len(frame_map):
            entries.append(list(UNSCORED_ENTRY))
        else:
            entries.append([source, frame_map[index]])
    return entries


def total_frames(segments):
    return sum(segment["frames"] for segment in segments)


def plan_frame_map(segments, sources):
    entries = []
    for segment in segments:
        entries.extend(segment_map_entries(segment, sources))
    return {"frames": entries, "empty_labelled": True}


def build_plan(name, segments, sources, fps, note=None):
    total = total_frames(segments)
    people = sum(s["frames"] for s in segments if s["kind"] != "empty")
    timeline = []
    cursor = 0
    for segment in segments:
        timeline.append({**segment, "start_s": cursor / fps, "duration_s": segment["frames"] / fps})
        cursor += segment["frames"]
    plan = {
        "name": name,
        "fps": fps,
        "total_frames": total,
        "duration_s": total / fps,
        "people_frames": people,
        "people_share": people / total if total else 0.0,
        "segments": timeline,
    }
    if note:
        plan["note"] = note
    return plan


def empty_segment(stretch, frames, mode="pingpong"):
    source, start, end = stretch
    return {"kind": "empty", "source": source, "start": start, "end": end, "frames": frames,
            "mode": mode}


def clip_segment(source, start, frames):
    return {"kind": "clip", "source": source, "start": start, "frames": frames}


def longest_run(runs, accept):
    candidates = [run for run in runs if accept(run)]
    if not candidates:
        raise ValueError("no hay un tramo con personas etiquetadas que cumpla la condición")
    return max(candidates, key=lambda run: (run[1] - run[0], -run[0]))


def plan_borde_inicio(labels, source, fps, clip_frames, lead_frames):
    runs = presence_runs(labels, MERGE_GAP)
    run = longest_run(runs, lambda r: r[1] - r[0] > lead_frames
                      and r[0] + lead_frames + clip_frames <= len(labels))
    return [clip_segment(source, run[0] + lead_frames, clip_frames)]


def plan_borde_final(labels, source, stretch, fps, empty_frames, lead_frames):
    runs = presence_runs(labels, MERGE_GAP)
    run = longest_run(runs, lambda r: r[0] >= lead_frames and r[1] - r[0] >= AFTER_FRAMES)
    last = run[0] + (run[1] - run[0]) // 2
    start = run[0] - lead_frames
    return [empty_segment(stretch, empty_frames),
            clip_segment(source, start, last - start + 1)]


def plan_borde_corto(labels, source, stretch, fps, empty_frames, window_frames):
    runs = presence_runs(labels, MERGE_GAP)
    run = longest_run(runs, lambda r: r[1] - r[0] + 1 >= window_frames
                      and all(labels[i] for i in range(r[0], r[1] + 1)))
    middle = run[0] + (run[1] - run[0]) // 2
    start = middle - window_frames // 2
    return [empty_segment(stretch, empty_frames),
            clip_segment(source, start, window_frames),
            empty_segment(stretch, empty_frames)]


def plan_borde_seguidos(labels_by_source, stretch, fps, window_frames, empty_frames,
                        lead_frames):
    best = None
    for source, labels in labels_by_source.items():
        if len(labels) < window_frames:
            continue
        start, count = densest_arrivals(first_appearances(labels), window_frames)
        if best is None or count > best[2]:
            best = (source, start, count)
    if best is None:
        raise ValueError("ninguna fuente alcanza para la ventana pedida")
    source, arrival, _ = best
    start = min(max(0, arrival - lead_frames), len(labels_by_source[source]) - window_frames)
    return [empty_segment(stretch, empty_frames),
            clip_segment(source, start, window_frames),
            empty_segment(stretch, empty_frames)]


def plan_borde_quieto(labels, source, stretch, fps, empty_frames, still_frames):
    runs = presence_runs(labels, MERGE_GAP)
    run = longest_run(runs, lambda r: any(len(labels[i] or []) == 1 for i in range(r[0], r[1] + 1)))
    middle = run[0] + (run[1] - run[0]) // 2
    singles = [i for i in range(run[0], run[1] + 1) if len(labels[i] or []) == 1]
    frame = min(singles, key=lambda i: abs(i - middle))
    return [empty_segment(stretch, empty_frames),
            {"kind": "repeat", "source": source, "start": frame, "frames": still_frames},
            empty_segment(stretch, empty_frames)]


def stretches_by_length(stretches):
    return sorted(stretches, key=lambda s: (s[2] - s[1], s[1]), reverse=True)


def plan_largo(stretches, clips, total_frames, fps, static_frames, weights, static_index,
               min_frames, max_frames):
    ordered = stretches_by_length(stretches)
    gaps = len(clips) + 1
    people = sum(clip["frames"] for clip in clips)
    lengths = distribute_empty(total_frames - people, gaps, static_index, static_frames,
                               weights, min_frames, max_frames)
    segments = []
    rotation = 0
    for gap in range(gaps):
        if gap == static_index:
            source, start, end = ordered[0]
            centre = start + (end - start) // 2
            segments.append(empty_segment((source, centre, centre), lengths[gap], "static"))
        else:
            segments.append(empty_segment(ordered[rotation % len(ordered)], lengths[gap]))
            rotation += 1
        if gap < len(clips):
            segments.append(clips[gap])
    return segments


def split_point(labels, margin):
    middle = len(labels) // 2
    free = [i for i in range(margin, len(labels) - margin)
            if not any(labels[i - margin:i + margin + 1])
            and all(label is not None for label in labels[i - margin:i + margin + 1])]
    if not free:
        return None
    return min(free, key=lambda i: abs(i - middle))


def load_json(path):
    with open(path) as f:
        return json.load(f)


def save_json(path, document):
    with open(path, "w") as f:
        json.dump(document, f)


def video_path(name):
    return os.path.join(chokepoint_dir(), "videos", f"{name}.mp4")


def load_source_gt(name):
    path = os.path.join(chokepoint_dir(), "groundtruth", f"{name}.xml")
    with open(path) as f:
        gt = metrics.parse_chokepoint_gt(f.read(), include_empty=True)
    frame_map = load_json(os.path.join(chokepoint_dir(), "videos", f"{name}.frames.json"))
    return gt, frame_map


def decode_source(name, cache_dir):
    os.makedirs(cache_dir, exist_ok=True)
    raw = os.path.join(cache_dir, f"{name}.raw")
    if not os.path.exists(raw):
        command = ["ffmpeg", "-v", "error", "-y", "-i", video_path(name), "-f", "rawvideo",
                   "-pix_fmt", "bgr24", raw + ".parcial"]
        subprocess.run(command, check=True)
        os.replace(raw + ".parcial", raw)
    frame_bytes = WIDTH * HEIGHT * 3
    count = os.path.getsize(raw) // frame_bytes
    return np.memmap(raw, dtype=np.uint8, mode="r", shape=(count, HEIGHT, WIDTH, 3))


def write_video(entries, stores, out_path, fps):
    command = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
               "-s", f"{WIDTH}x{HEIGHT}", "-r", str(fps), "-i", "-", "-c:v", "libx264",
               "-crf", "18", "-pix_fmt", "yuv420p", out_path]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    try:
        for source, index in entries:
            process.stdin.write(stores[source][index].tobytes())
    finally:
        process.stdin.close()
    if process.wait() != 0:
        raise SystemExit(f"ffmpeg falló al escribir {out_path}")


def source_infos(names, cache_dir):
    stores, infos, labels = {}, {}, {}
    for name in names:
        store = decode_source(name, cache_dir)
        stores[name] = store
        has_gt = os.path.exists(os.path.join(chokepoint_dir(), "videos", f"{name}.frames.json"))
        if has_gt:
            gt, frame_map = load_source_gt(name)
            infos[name] = {"frames": len(store), "map": frame_map}
            labels[name] = source_labels(gt, frame_map, len(store))
        else:
            infos[name] = {"frames": len(store), "map": None}
    return stores, infos, labels


def run_checks(name, out_dir):
    path = os.path.join(out_dir, name + CHECKS_SUFFIX)
    if os.path.exists(path):
        return load_json(path)["checks"]
    command = [os.path.join(pipeline_dir(), "venvs", "track", "bin", "python"),
               os.path.join(os.path.dirname(os.path.abspath(__file__)), "chequeo_personas.py"),
               "--video", video_path(name), "--step", str(CHECK_STEP), "--conf", str(CHECK_CONF),
               "--imgsz", str(CHECK_IMGSZ),
               "--weights", os.path.join(pipeline_dir(), "yolo26m.pt"), "--out", path]
    subprocess.run(command, check=True)
    return load_json(path)["checks"]


def command_vacios(args):
    out_dir = args.salida
    os.makedirs(out_dir, exist_ok=True)
    cache_dir = os.path.join(out_dir, "cache")
    _, _, labels = source_infos(LABELLED_SOURCES, cache_dir)
    margin = round(EMPTY_MARGIN_S * FPS)
    found = []
    for name in LABELLED_SOURCES:
        checks = run_checks(name, out_dir)
        for start, end in empty_stretches(labels[name], checks, CHECK_STEP, margin,
                                          round(EMPTY_MIN_S * FPS)):
            found.append({"source": name, "start": start, "end": end,
                          "start_s": start / FPS, "end_s": (end + 1) / FPS,
                          "duration_s": (end - start + 1) / FPS})
    document = {"fps": FPS, "stretches": found,
                "total_s": sum(item["duration_s"] for item in found)}
    save_json(os.path.join(out_dir, EMPTY_NAME), document)
    for item in found:
        print(f"{item['source']}: fotogramas {item['start']} a {item['end']} "
              f"({item['duration_s']:.1f} s)")
    print(f"total de metraje vacío: {document['total_s']:.1f} s en {len(found)} tramos")


def load_stretches(out_dir):
    path = os.path.join(out_dir, EMPTY_NAME)
    if not os.path.exists(path):
        raise SystemExit(f"falta {path}: ejecuta primero el subcomando vacios")
    stretches = [(item["source"], item["start"], item["end"])
                 for item in load_json(path)["stretches"]]
    if not stretches:
        raise SystemExit("no hay tramos vacíos")
    return stretches


def print_plan(plan):
    print(f"plan {plan['name']}: {plan['total_frames']} fotogramas, {plan['duration_s']:.1f} s, "
          f"personas {plan['people_share']:.1%}")
    for segment in plan["segments"]:
        print(f"  {segment['start_s']:9.1f} s  {segment['duration_s']:8.1f} s  "
              f"{segment['kind']:6s} {segment['source']} desde {segment['start']}"
              f"{' (' + segment['mode'] + ')' if 'mode' in segment else ''}")


def emit(name, segments, sources, stores, out_dir, note=None):
    plan = build_plan(name, segments, sources, FPS, note)
    frame_map = plan_frame_map(segments, sources)
    entries = []
    for segment in segments:
        entries.extend(expand_segment(segment))
    if len(entries) != len(frame_map["frames"]):
        raise SystemExit("el mapa y los fotogramas no coinciden")
    print_plan(plan)
    write_video(entries, stores, os.path.join(out_dir, f"{name}.mp4"), FPS)
    save_json(os.path.join(out_dir, f"{name}.frames.json"), frame_map)
    save_json(os.path.join(out_dir, f"{name}.plan.json"), plan)


def halves(name, labels):
    point = split_point(labels, FPS // 2)
    frames = len(labels)
    if point is None:
        return [clip_segment(name, 0, frames)]
    return [clip_segment(name, 0, point), clip_segment(name, point, frames - point)]


def command_largo(args):
    out_dir = args.salida
    stretches = load_stretches(out_dir)
    stores, sources, labels = source_infos(LABELLED_SOURCES + [BUSY_SOURCE],
                                           os.path.join(out_dir, "cache"))
    first = halves(LABELLED_SOURCES[0], labels[LABELLED_SOURCES[0]])
    second = halves(LABELLED_SOURCES[1], labels[LABELLED_SOURCES[1]])
    busy = clip_segment(BUSY_SOURCE, 0, BUSY_CLIP_S * FPS)
    clips = first + second + [busy]
    weights = LARGE_WEIGHTS[:len(clips) + 1]
    segments = plan_largo(stretches, clips, args.minutos * 60 * FPS, FPS,
                          LARGE_STATIC_S * FPS, weights, LARGE_STATIC_INDEX,
                          LARGE_MIN_S * FPS, LARGE_MAX_S * FPS)
    emit("largo", segments, sources, stores, out_dir,
         "metraje vacío en ping-pong; un tramo estático de un solo fotograma; "
         "20 s de P2E_S5_C1 sin etiquetas y sin puntuar")


def command_bordes(args):
    out_dir = args.salida
    stretches = stretches_by_length(load_stretches(out_dir))
    stretch = stretches[0]
    stores, sources, labels = source_infos(LABELLED_SOURCES, os.path.join(out_dir, "cache"))
    primary = LABELLED_SOURCES[0]
    plans = {
        "borde_inicio": (plan_borde_inicio(labels[primary], primary, FPS, 20 * FPS, FPS), None),
        "borde_final": (plan_borde_final(labels[primary], primary, stretch, FPS, 30 * FPS, FPS),
                        None),
        "borde_corto": (plan_borde_corto(labels[primary], primary, stretch, FPS, 15 * FPS,
                                         round(0.6 * FPS)), None),
        "borde_seguidos": (plan_borde_seguidos(labels, stretch, FPS, 20 * FPS, 10 * FPS,
                                               FPS // 2), None),
        "borde_quieto": (plan_borde_quieto(labels[primary], primary, stretch, FPS, 10 * FPS,
                                           60 * FPS),
                         "quietud artificial: un solo fotograma etiquetado repetido 60 s"),
    }
    for name, (segments, note) in plans.items():
        emit(name, segments, sources, stores, out_dir, note)


def build_parser():
    parser = argparse.ArgumentParser(description="Construye el video largo de prueba y los "
                                                 "clips de borde con sus mapas de fotogramas")
    parser.add_argument("--salida", default=default_output_dir(),
                        help="carpeta de salida (videos, mapas, planes)")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("vacios", help="busca los tramos sin personas en las fuentes etiquetadas")
    largo = sub.add_parser("largo", help="construye el video largo")
    largo.add_argument("--minutos", type=int, default=45, help="duración total en minutos")
    sub.add_parser("bordes", help="construye los clips de borde")
    return parser


def main():
    args = build_parser().parse_args()
    {"vacios": command_vacios, "largo": command_largo, "bordes": command_bordes}[args.command](args)


if __name__ == "__main__":
    main()
