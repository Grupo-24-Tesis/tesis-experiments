import argparse
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))

from bench import bench_compuerta, metrics

GATE_FLOOR_CONF = "0.05"
GATE_TIMES_NAME = "gate_times.json"
SETTING_FOLDER = re.compile(r"^c([0-9.]+)_i([0-9]+)$")
MAX_RECALL_DROP_POINTS = 0.5
MAX_VIDEO_DROP_POINTS = 1.0
DEFAULT_OVERHEAD_S = 20.0
RECALL_NOTE = ("La exhaustividad simulada se mide sobre los fotogramas analizados de la base: "
               "una persona etiquetada en un fotograma que la compuerta omite cuenta como "
               "no encontrada")
ESTIMATE_NOTE = ("El tiempo de rostros es una estimación: la parte del tiempo de la base que "
                 "supera la sobrecarga fija de carga de modelos (--overhead-s) se escala por la "
                 "razón entre fotogramas analizados simulados y de la base, y la sobrecarga se "
                 "suma una vez por video")
MISSING_CONFIDENCES_NOTE = ("AVISO: presence.json sin confidences; se toma cada revisión "
                            "encontrada como confianza 1.0 y cada no encontrada como 0.0")


def load_active_intervals():
    pipeline = bench_compuerta.pipeline_dir()
    if pipeline not in sys.path:
        sys.path.insert(0, pipeline)
    from presence_intervals import active_intervals, switch_log
    return active_intervals, switch_log


def setting_folder(checks, imgsz):
    return f"c{checks:g}_i{imgsz}"


def parse_number_list(text, kind, name):
    values = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            values.append(kind(part))
        except ValueError:
            raise SystemExit(f"{name} espera una lista separada por comas, no {text!r}")
    if not values:
        raise SystemExit(f"{name} no puede estar vacío")
    return values


def parse_name_list(text):
    if text is None:
        return None
    return [part.strip() for part in text.split(",") if part.strip()]


def read_presence(path):
    try:
        return bench_compuerta.load_json(path)
    except (OSError, ValueError):
        return None


def read_gate_times(out_dir):
    path = os.path.join(out_dir, GATE_TIMES_NAME)
    if os.path.exists(path):
        return bench_compuerta.load_json(path)
    return {}


def write_gate_times(out_dir, times):
    with open(os.path.join(out_dir, GATE_TIMES_NAME), "w") as f:
        json.dump(times, f, indent=2)


def command_gate(args):
    checks_list = parse_number_list(args.checks_per_s, float, "--checks-per-s")
    imgsz_list = parse_number_list(args.imgsz, int, "--imgsz")
    os.makedirs(args.out, exist_ok=True)
    times = read_gate_times(args.out)
    pipeline = bench_compuerta.pipeline_dir()
    interpreter = os.path.join(pipeline, "venvs", "track", "bin", "python")
    failures = 0
    for checks in checks_list:
        for imgsz in imgsz_list:
            folder = setting_folder(checks, imgsz)
            env = {**os.environ, "GATE_CONF": GATE_FLOOR_CONF, "GATE_CHECKS_PER_S": f"{checks:g}",
                   "GATE_IMGSZ": str(imgsz)}
            for video in args.videos:
                name = bench_compuerta.video_name(video)
                video_dir = os.path.join(args.out, folder, name)
                presence_path = os.path.join(video_dir, bench_compuerta.PRESENCE_NAME)
                if read_presence(presence_path) is not None:
                    print(f"[{folder}/{name}] ya existe, se omite")
                    continue
                os.makedirs(video_dir, exist_ok=True)
                partial_path = presence_path + ".parcial"
                command = [interpreter, "presence_gate.py", "--video", os.path.abspath(video),
                           "--out", partial_path]
                wall, code = bench_compuerta.run_logged(f"{folder}/{name}", command, env,
                                                        os.path.join(video_dir, "gate.log"))
                if code == 0 and read_presence(partial_path) is None:
                    code = 1
                if code != 0:
                    for stale in (partial_path, presence_path):
                        if os.path.exists(stale):
                            os.remove(stale)
                    print(f"[{folder}/{name}] la compuerta falló con código {code}, "
                          f"ver {video_dir}/gate.log")
                    failures += 1
                    continue
                os.replace(partial_path, presence_path)
                times.setdefault(folder, {})[name] = wall
                write_gate_times(args.out, times)
                print(f"[{folder}/{name}] compuerta {wall:.1f} s")
    return 1 if failures else 0


def retained_frames(frames, intervals):
    kept = []
    position = 0
    for frame in sorted(frames):
        while position < len(intervals) and intervals[position][1] < frame:
            position += 1
        if position < len(intervals) and intervals[position][0] <= frame:
            kept.append(frame)
    return kept


def checks_at(presence, conf):
    confidences = presence.get("confidences")
    if confidences is None:
        return list(presence["checks"]), True
    return [[frame, value > conf] for frame, value in confidences], False


def simulate_intervals(presence, conf, exit_s, rewind_s):
    active_intervals, _ = load_active_intervals()
    checks, missing = checks_at(presence, conf)
    intervals = active_intervals(checks, presence["fps"], presence["total_frames"],
                                 rewind_s, exit_s)
    return [tuple(interval) for interval in intervals], missing


def switch_frames(intervals, total_frames):
    frames = []
    for start, end in intervals:
        frames.append(start)
        if total_frames <= 0 or end + 1 < total_frames:
            frames.append(end + 1)
    return frames


def lost_detection_records(detections, kept_frames, intervals, total_frames, fps):
    kept = set(kept_frames)
    switches = switch_frames(intervals, total_frames)
    records = []
    for detection in detections:
        if detection["frame"] in kept:
            continue
        distance = min((abs(detection["frame"] - frame) for frame in switches), default=None)
        records.append({"frame": detection["frame"], "time_s": detection["time_s"],
                        "box": detection["box"],
                        "distance_s": None if distance is None else distance / fps})
    return records


def person_true_positive_frames(detections, frames, gt, frame_map):
    by_frame = {}
    for detection in detections:
        by_frame.setdefault(detection["frame"], []).append(detection["box"])
    found = {}
    for frame in frames:
        people = metrics.people_at(gt, frame_map, frame)
        if people is None:
            continue
        boxes = by_frame.get(frame, [])
        for pid, point in people:
            if sum(1 for box in boxes if metrics.inside(box, point)) == 1:
                found.setdefault(pid, set()).add(frame)
    return found


def persons_fully_lost(person_frames, kept_frames):
    kept = set(kept_frames)
    return sorted(pid for pid, frames in person_frames.items() if frames and not (frames & kept))


def detection_metrics(case, detections, frames):
    return bench_compuerta.detection_metrics(case["kind"], case["gt"], case["frame_map"],
                                             case["fps"], detections, frames)


def build_case(name, entry, detections_doc):
    kind, gt, frame_map = bench_compuerta.find_ground_truth(name)
    case = {"name": name, "kind": kind, "gt": gt, "frame_map": frame_map,
            "fps": detections_doc["fps"], "frames": sorted(detections_doc["frames_analysed"]),
            "detections": detections_doc["detections"],
            "base_wall_s": entry.get("face_wall_s", entry["wall_s"])}
    case["base_detection"] = detection_metrics(case, case["detections"], case["frames"])
    case["person_frames"] = (person_true_positive_frames(case["detections"], case["frames"],
                                                         gt, frame_map)
                             if kind == bench_compuerta.KIND_CHOKEPOINT else {})
    return case


def estimate_face_s(base_wall_s, kept, analysed, overhead_s):
    if not analysed:
        return 0.0
    overhead = min(overhead_s, base_wall_s)
    return overhead + (base_wall_s - overhead) * kept / analysed


def score_video(case, presence, conf, exit_s, rewind_s, overhead_s=DEFAULT_OVERHEAD_S):
    intervals, missing = simulate_intervals(presence, conf, exit_s, rewind_s)
    kept = retained_frames(case["frames"], intervals)
    kept_set = set(kept)
    detections = [d for d in case["detections"] if d["frame"] in kept_set]
    detection = detection_metrics(case, detections, case["frames"])
    base = case["base_detection"]
    analysed = len(case["frames"])
    result = {
        "analysed_frames": len(kept),
        "analysed_fraction": len(kept) / analysed if analysed else 0.0,
        "estimated_face_s": estimate_face_s(case["base_wall_s"], len(kept), analysed,
                                            overhead_s),
        "lost": lost_detection_records(case["detections"], kept, intervals,
                                       presence["total_frames"], case["fps"]),
        "persons_lost": persons_fully_lost(case["person_frames"], kept),
        "confidences_missing": missing,
        "detection": detection,
        "base_detection": base,
        "recall_drop_points": 0.0,
    }
    if detection and base:
        result["recall_drop_points"] = (base["recall"] - detection["recall"]) * 100
    return result


def summarise_setting(per_video, cases, precision_names, cost_names, gate_times):
    precision = [n for n in per_video if n in precision_names and per_video[n]["detection"]]
    base_tp = sum(per_video[n]["base_detection"]["tp"] for n in precision)
    base_fn = sum(per_video[n]["base_detection"]["fn"] for n in precision)
    sim_tp = sum(per_video[n]["detection"]["tp"] for n in precision)
    sim_fn = sum(per_video[n]["detection"]["fn"] for n in precision)
    base_recall = base_tp / (base_tp + base_fn) if base_tp + base_fn else 0.0
    sim_recall = sim_tp / (sim_tp + sim_fn) if sim_tp + sim_fn else 0.0
    cost = [n for n in per_video if n in cost_names]
    gate_total = sum(gate_times.get(n, 0.0) for n in cost)
    face_total = sum(per_video[n]["estimated_face_s"] for n in cost)
    base_total = sum(cases[n]["base_wall_s"] for n in cost)
    estimated_total = gate_total + face_total
    return {
        "videos_precision": precision,
        "videos_cost": cost,
        "recall_change_points": (sim_recall - base_recall) * 100,
        "videos_with_drop": sum(1 for n in precision
                                if per_video[n]["recall_drop_points"] > 1e-9),
        "max_video_drop_points": max([per_video[n]["recall_drop_points"] for n in precision],
                                     default=0.0),
        "persons_lost": sum(len(per_video[n]["persons_lost"]) for n in precision),
        "mean_analysed_fraction": (sum(per_video[n]["analysed_fraction"] for n in cost)
                                   / len(cost)) if cost else 0.0,
        "gate_s": gate_total,
        "estimated_face_s": face_total,
        "estimated_total_s": estimated_total,
        "base_total_s": base_total,
        "factor": estimated_total / base_total if base_total else None,
    }


def is_safe(summary):
    return (summary["recall_change_points"] >= -MAX_RECALL_DROP_POINTS
            and summary["persons_lost"] == 0
            and summary["max_video_drop_points"] <= MAX_VIDEO_DROP_POINTS)


def recommend(settings):
    safe = [s for s in settings if not s["missing"] and s["summary"]["factor"] is not None
            and is_safe(s["summary"])]
    if not safe:
        return None
    return min(safe, key=lambda s: (s["summary"]["factor"], -s["exit_s"], -s["rewind_s"]))


def discover_gates(gates_dir):
    found = {}
    for folder in sorted(os.listdir(gates_dir)):
        match = SETTING_FOLDER.match(folder)
        if match and os.path.isdir(os.path.join(gates_dir, folder)):
            found[folder] = (float(match.group(1)), int(match.group(2)))
    return found


def load_baselines(paths):
    cases = {}
    for path in paths:
        document = bench_compuerta.load_json(path)
        base_dir = os.path.dirname(os.path.abspath(path))
        for name, entry in document["videos"].items():
            if "error" in entry or name in cases:
                continue
            detections_doc = bench_compuerta.load_json(
                os.path.join(base_dir, entry["detections_file"]))
            cases[name] = build_case(name, entry, detections_doc)
    return cases


def sweep(cases, gates_dir, confs, exits, rewinds, precision_names, cost_names,
          overhead_s=DEFAULT_OVERHEAD_S):
    gate_folders = discover_gates(gates_dir)
    times = read_gate_times(gates_dir)
    settings = []
    missing_notes = set()
    for folder, (checks, imgsz) in gate_folders.items():
        presences = {}
        for name in cases:
            path = os.path.join(gates_dir, folder, name, bench_compuerta.PRESENCE_NAME)
            presence = read_presence(path)
            if presence is not None:
                presences[name] = presence
        if not presences:
            continue
        missing = sorted((precision_names | cost_names) - set(presences))
        gate_times = {n: times.get(folder, {}).get(n, p.get("wall_s", 0.0))
                      for n, p in presences.items()}
        for conf in confs:
            for exit_s in exits:
                for rewind_s in rewinds:
                    per_video = {}
                    for name, presence in presences.items():
                        per_video[name] = score_video(cases[name], presence, conf, exit_s,
                                                      rewind_s, overhead_s)
                        if per_video[name]["confidences_missing"]:
                            missing_notes.add(f"{folder}/{name}")
                    summary = summarise_setting(per_video, cases, precision_names, cost_names,
                                                gate_times)
                    settings.append({"checks_per_s": checks, "imgsz": imgsz, "conf": conf,
                                     "exit_s": exit_s, "rewind_s": rewind_s,
                                     "missing": missing, "summary": summary,
                                     "per_video": per_video})
    return settings, sorted(missing_notes)


def format_table(settings, recommended):
    header = ["rev/s", "imgsz", "conf", "salida", "retroceso", "exhaust.", "videos con caída",
              "personas perdidas", "peor video", "fracción", "compuerta s", "total est. s",
              "factor", ""]
    rows = [header]
    for setting in settings:
        s = setting["summary"]
        rows.append([f"{setting['checks_per_s']:g}", str(setting["imgsz"]),
                     f"{setting['conf']:g}", f"{setting['exit_s']:g}",
                     f"{setting['rewind_s']:g}", f"{s['recall_change_points']:+.2f} pp",
                     str(s["videos_with_drop"]), str(s["persons_lost"]),
                     f"{s['max_video_drop_points']:.2f} pp",
                     f"{s['mean_analysed_fraction']:.1%}", f"{s['gate_s']:.1f}",
                     f"{s['estimated_total_s']:.1f}",
                     "n/d" if s["factor"] is None else f"{s['factor']:.2f}",
                     "RECOMENDADO" if setting is recommended
                     else (f"INCOMPLETO (faltan {', '.join(setting['missing'])})"
                           if setting["missing"] else "")])
    widths = [max(len(row[i]) for row in rows) for i in range(len(header))]
    return ["  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)).rstrip()
            for row in rows]


def command_puntuar(args):
    confs = parse_number_list(args.conf, float, "--conf")
    exits = parse_number_list(args.exit, float, "--exit")
    rewinds = parse_number_list(args.rewind, float, "--rewind")
    cases = load_baselines(args.baseline)
    if not os.path.isdir(args.gates):
        raise SystemExit(f"No existe la carpeta de compuertas: {args.gates}")
    precision_names = parse_name_list(args.videos_precision)
    cost_names = parse_name_list(args.videos_coste)
    precision_names = set(cases) if precision_names is None else set(precision_names)
    cost_names = set(cases) if cost_names is None else set(cost_names)
    unknown = sorted((precision_names | cost_names) - set(cases))
    if unknown:
        raise SystemExit(f"Videos que no están en la base: {', '.join(unknown)}")
    settings, missing = sweep(cases, args.gates, confs, exits, rewinds, precision_names,
                              cost_names, args.overhead_s)
    if not settings:
        raise SystemExit("No hay compuertas que coincidan con videos de la base")
    recommended = recommend(settings)
    lines = format_table(settings, recommended)
    if missing:
        lines.append(MISSING_CONFIDENCES_NOTE)
    lines.append(RECALL_NOTE)
    lines.append(ESTIMATE_NOTE)
    lines.append(f"Regla de recomendación: tiempo total estimado mínimo con caída total de "
                 f"exhaustividad <= {MAX_RECALL_DROP_POINTS} pp, ninguna persona perdida y "
                 f"ningún video con caída mayor que {MAX_VIDEO_DROP_POINTS} pp")
    if recommended is None:
        lines.append("Ningún ajuste cumple la regla de recomendación")
    print("\n".join(lines))
    document = {"created": time.strftime("%Y-%m-%dT%H:%M:%S"), "estimate_note": ESTIMATE_NOTE,
                "recall_note": RECALL_NOTE,
                "overhead_s": args.overhead_s,
                "baselines": args.baseline, "gates": args.gates,
                "videos_precision": sorted(precision_names),
                "videos_cost": sorted(cost_names),
                "confidences_missing": missing,
                "settings": settings,
                "recommended": None if recommended is None else {
                    key: recommended[key] for key in
                    ("checks_per_s", "imgsz", "conf", "exit_s", "rewind_s")}}
    with open(args.out, "w") as f:
        json.dump(document, f, indent=2)
    print(f"Resultados escritos en {args.out}")
    return 0


def build_parser():
    parser = argparse.ArgumentParser(
        description="Barrido de umbrales de la compuerta de presencia: corre solo la pasada de "
                    "la compuerta por ajuste y puntúa el resto sin conexión desde la base")
    sub = parser.add_subparsers(dest="command", required=True)
    gate = sub.add_parser("gate", help="corre presence_gate.py por video y ajuste")
    gate.add_argument("--out", required=True, help="carpeta de salida DIR/c<rev>_i<imgsz>/VIDEO")
    gate.add_argument("--checks-per-s", required=True,
                      help="revisiones por segundo, separadas por comas")
    gate.add_argument("--imgsz", required=True,
                      help="tamaños de imagen de YOLO, separados por comas")
    gate.add_argument("videos", nargs="+", metavar="VIDEO", help="videos a revisar")
    gate.set_defaults(handler=command_gate)
    score = sub.add_parser("puntuar", help="puntúa los ajustes sin conexión desde la base")
    score.add_argument("--baseline", action="append", required=True, metavar="RESULTS_JSON",
                       help="results.json de la base sin compuerta (se puede repetir)")
    score.add_argument("--gates", required=True, help="carpeta escrita por el subcomando gate")
    score.add_argument("--conf", required=True, help="umbrales de confianza, separados por comas")
    score.add_argument("--exit", required=True, help="segundos de salida, separados por comas")
    score.add_argument("--rewind", required=True, help="segundos de retroceso, separados por comas")
    score.add_argument("--out", required=True, help="archivo JSON de salida")
    score.add_argument("--overhead-s", type=float, default=DEFAULT_OVERHEAD_S, metavar="S",
                       help="segundos fijos por video (carga de modelos) que no se escalan con "
                            "los fotogramas analizados (por defecto 20)")
    score.add_argument("--videos-coste", default=None, metavar="NOMBRE,...",
                       help="videos que cuentan para las columnas de costo (por defecto todos)")
    score.add_argument("--videos-precision", default=None, metavar="NOMBRE,...",
                       help="videos que cuentan para las columnas de precisión (por defecto "
                            "todos)")
    score.set_defaults(handler=command_puntuar)
    return parser


def main():
    args = build_parser().parse_args()
    sys.exit(args.handler(args))


if __name__ == "__main__":
    main()
