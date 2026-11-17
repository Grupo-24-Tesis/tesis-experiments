import argparse
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))

from bench import metrics

EXPERIMENTS_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
DEFAULT_PIPELINE_DIR = "/data/TesisFaceRecognition/wt/tesis_face_recognition_pipeline"
DEFAULT_CHOKEPOINT_DIR = "/data/TesisFaceRecognition/chokepoint"
RESULTS_NAME = "results.json"
DETECTIONS_NAME = "detections.json"
IDENTITIES_NAME = "identities.json"
KIND_CHOKEPOINT = "chokepoint"
KIND_CCTV = "cctv"
DEFAULT_MAX_RECALL_DROP = 0.5


def pipeline_dir():
    return os.environ.get("TESIS_PIPELINE_DIR", DEFAULT_PIPELINE_DIR)


def chokepoint_dir():
    return os.environ.get("TESIS_CHOKEPOINT_DIR", DEFAULT_CHOKEPOINT_DIR)


def video_name(path):
    return os.path.splitext(os.path.basename(path))[0]


def load_json(path):
    with open(path) as f:
        return json.load(f)


def find_ground_truth(name):
    root = chokepoint_dir()
    xml_path = os.path.join(root, "groundtruth", f"{name}.xml")
    map_path = os.path.join(root, "videos", f"{name}.frames.json")
    if os.path.exists(xml_path) and os.path.exists(map_path):
        with open(xml_path) as f:
            gt = metrics.parse_chokepoint_gt(f.read())
        return KIND_CHOKEPOINT, gt, load_json(map_path)
    cctv_path = os.path.join(EXPERIMENTS_DIR, "ground_truth", name, "frames.json")
    if os.path.exists(cctv_path):
        return KIND_CCTV, metrics.parse_cctv_gt(load_json(cctv_path)), None
    return None, None, None


def evaluate_video(name, identities_doc, detections_doc):
    kind, gt, frame_map = find_ground_truth(name)
    result = {"kind": kind, "identity": None, "detection": None}
    identities = identities_doc["identities"]
    detections = detections_doc["detections"]
    analysed = detections_doc["frames_analysed"]
    if kind == KIND_CHOKEPOINT:
        result["identity"] = metrics.chokepoint_identity_metrics(identities, gt, frame_map)
        result["detection"] = metrics.chokepoint_detection_metrics(detections, analysed, gt,
                                                                   frame_map)
    elif kind == KIND_CCTV:
        fps = detections_doc["fps"]
        result["identity"] = metrics.cctv_identity_metrics(identities, fps, gt)
        result["detection"] = metrics.cctv_detection_metrics(detections, analysed, fps, gt)
    return result


def video_duration(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                          "-of", "default=noprint_wrappers=1:nokey=1", path],
                         capture_output=True, text=True)
    try:
        return float(out.stdout.strip())
    except ValueError:
        return None


def parse_env(pairs):
    env = {}
    for pair in pairs:
        key, separator, value = pair.partition("=")
        if not separator or not key:
            raise SystemExit(f"--env espera CLAVE=VALOR, no {pair!r}")
        env[key] = value
    return env


def run_video(video, label_dir, env_overrides, extra_args):
    name = video_name(video)
    faces_dir = os.path.join(label_dir, name)
    os.makedirs(faces_dir, exist_ok=True)
    pipeline = pipeline_dir()
    python = os.path.join(pipeline, "venvs", "faces", "bin", "python")
    command = [python, "face_identities.py", "--video", video, "--faces-dir", faces_dir,
               *extra_args]
    env = {**os.environ, **env_overrides}
    print(f"[{name}] ejecutando: {' '.join(command)}", flush=True)
    started = time.monotonic()
    with open(os.path.join(faces_dir, "run.log"), "w") as log:
        completed = subprocess.run(command, cwd=pipeline, env=env, stdout=log,
                                   stderr=subprocess.STDOUT)
    wall = time.monotonic() - started
    if completed.returncode != 0:
        print(f"[{name}] fallo con código {completed.returncode}, ver {faces_dir}/run.log")
        return {"error": f"código de salida {completed.returncode}", "wall_s": wall}
    identities_doc = load_json(os.path.join(faces_dir, IDENTITIES_NAME))
    detections_doc = load_json(os.path.join(faces_dir, DETECTIONS_NAME))
    duration = video_duration(video)
    entry = {
        "video": video,
        "duration_s": duration,
        "wall_s": wall,
        "factor": wall / duration if duration else None,
        "analysed_frames": len(detections_doc["frames_analysed"]),
        "detections_count": len(detections_doc["detections"]),
        "identities_count": len(identities_doc["identities"]),
        "detections_file": os.path.join(name, DETECTIONS_NAME),
    }
    entry.update(evaluate_video(name, identities_doc, detections_doc))
    describe(name, entry)
    return entry


def describe(name, entry):
    factor = entry["factor"]
    print(f"[{name}] {entry['wall_s']:.1f} s"
          + (f", factor {factor:.2f}" if factor is not None else "")
          + f", {entry['analysed_frames']} fotogramas, {entry['detections_count']} detecciones, "
            f"{entry['identities_count']} identidades")
    identity, detection = entry.get("identity"), entry.get("detection")
    if identity:
        print(f"[{name}] etiquetadas {identity['labelled']} | encontradas {identity['found']} | "
              f"duplicados {identity['duplicates']} | mezcladas {identity['mixed']} | "
              f"pureza {identity['purity']:.1%}")
    if detection:
        print(f"[{name}] detección: precisión {detection['precision']:.1%} | "
              f"exhaustividad {detection['recall']:.1%} | F1 {detection['f1']:.1%} "
              f"(TP {detection['tp']}, FP {detection['fp']}, FN {detection['fn']})")


def command_run(args):
    label_dir = os.path.join(args.out, args.label)
    os.makedirs(label_dir, exist_ok=True)
    env_overrides = parse_env(args.env)
    videos = {}
    for video in args.videos:
        videos[video_name(video)] = run_video(os.path.abspath(video), label_dir, env_overrides,
                                              args.extra_arg)
    document = {"label": args.label, "env": env_overrides, "extra_args": args.extra_arg,
                "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "videos": videos}
    results_path = os.path.join(label_dir, RESULTS_NAME)
    with open(results_path, "w") as f:
        json.dump(document, f, indent=2)
    print(f"Resultados escritos en {results_path}")
    return 1 if any("error" in entry for entry in videos.values()) else 0


def ratio(candidate, baseline):
    if candidate is None or baseline in (None, 0):
        return None
    return candidate / baseline


def points(candidate, baseline):
    return (candidate - baseline) * 100


def fmt_ratio(value):
    return "n/d" if value is None else f"{value:.2f}"


def fmt_pct(value):
    return f"{value:.1%}"


def fmt_points(value):
    return f"{value:+.1f} pp"


def totals_recall(videos):
    tp = sum(v["detection"]["tp"] for v in videos.values() if v.get("detection"))
    fn = sum(v["detection"]["fn"] for v in videos.values() if v.get("detection"))
    return tp / (tp + fn) if tp + fn else None


def compare_documents(baseline, candidate, max_recall_drop, load_detections):
    lines = []
    failures = []
    names = [n for n in baseline["videos"] if n in candidate["videos"]]
    for name in names:
        base, cand = baseline["videos"][name], candidate["videos"][name]
        lines.append(f"Video {name}")
        if "error" in base or "error" in cand:
            lines.append("  sin resultados comparables (una de las ejecuciones falló)")
            failures.append(f"{name}: una de las ejecuciones falló")
            continue
        lines.append(f"  factor de tiempo: {fmt_ratio(base['factor'])} -> {fmt_ratio(cand['factor'])}")
        lines.append(f"  fotogramas analizados: {base['analysed_frames']} -> "
                     f"{cand['analysed_frames']} (razón {fmt_ratio(ratio(cand['analysed_frames'], base['analysed_frames']))})")
        if base.get("detection") and cand.get("detection"):
            bd, cd = base["detection"], cand["detection"]
            for key, title in (("precision", "precisión"), ("recall", "exhaustividad"),
                               ("f1", "F1")):
                lines.append(f"  {title}: {fmt_pct(bd[key])} -> {fmt_pct(cd[key])} "
                             f"({fmt_points(points(cd[key], bd[key]))})")
        if base.get("identity") and cand.get("identity"):
            bi, ci = base["identity"], cand["identity"]
            lines.append(f"  personas encontradas: {bi['found']} -> {ci['found']} de {bi['labelled']}")
            lines.append(f"  duplicados: {bi['duplicates']} -> {ci['duplicates']}")
            lines.append(f"  mezcladas: {bi['mixed']} -> {ci['mixed']}")
            lines.append(f"  pureza: {fmt_pct(bi['purity'])} -> {fmt_pct(ci['purity'])}")
            if ci["found"] < bi["found"]:
                failures.append(f"{name}: personas encontradas {bi['found']} -> {ci['found']}")
        missing = metrics.missing_detections(load_detections(baseline, base),
                                             load_detections(candidate, cand))
        lines.append(f"  detecciones de la base ausentes en el candidato: {len(missing)}")
        for detection in missing:
            lines.append(f"    fotograma {detection['frame']} | "
                         f"{metrics.format_time(detection['time_s'])} | caja {detection['box']}")
    lines.append("Totales")
    base_videos = {n: baseline["videos"][n] for n in names if "error" not in baseline["videos"][n]}
    cand_videos = {n: candidate["videos"][n] for n in names if "error" not in candidate["videos"][n]}
    for title, key in (("tiempo de pared", "wall_s"), ("duración", "duration_s")):
        lines.append(f"  {title}: {sum(v[key] or 0 for v in base_videos.values()):.1f} s -> "
                     f"{sum(v[key] or 0 for v in cand_videos.values()):.1f} s")
    base_duration = sum(v["duration_s"] or 0 for v in base_videos.values())
    cand_duration = sum(v["duration_s"] or 0 for v in cand_videos.values())
    if base_duration and cand_duration:
        lines.append(f"  factor de tiempo: "
                     f"{sum(v['wall_s'] for v in base_videos.values()) / base_duration:.2f} -> "
                     f"{sum(v['wall_s'] for v in cand_videos.values()) / cand_duration:.2f}")
    lines.append(f"  fotogramas analizados: "
                 f"{sum(v['analysed_frames'] for v in base_videos.values())} -> "
                 f"{sum(v['analysed_frames'] for v in cand_videos.values())}")
    base_recall, cand_recall = totals_recall(base_videos), totals_recall(cand_videos)
    if base_recall is not None and cand_recall is not None:
        drop = points(base_recall, cand_recall)
        lines.append(f"  exhaustividad total: {fmt_pct(base_recall)} -> {fmt_pct(cand_recall)} "
                     f"({fmt_points(points(cand_recall, base_recall))})")
        if drop > max_recall_drop:
            failures.append(f"exhaustividad total baja {drop:.1f} pp, más del máximo de "
                            f"{max_recall_drop:.1f} pp")
    for failure in failures:
        lines.append(f"REGRESIÓN: {failure}")
    return lines, 1 if failures else 0


def command_compare(args):
    baseline = load_json(args.baseline)
    candidate = load_json(args.candidate)

    def load_detections(document, entry):
        base_dir = args.baseline if document is baseline else args.candidate
        path = os.path.join(os.path.dirname(os.path.abspath(base_dir)), entry["detections_file"])
        return load_json(path)["detections"]

    lines, code = compare_documents(baseline, candidate, args.max_recall_drop, load_detections)
    print("\n".join(lines))
    return code


def build_parser():
    parser = argparse.ArgumentParser(
        description="Banco de pruebas de la compuerta de presencia: ejecuta face_identities.py "
                    "sobre videos, mide tiempo, detección e identidades y compara ejecuciones")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="ejecuta el pipeline sobre los videos y guarda results.json")
    run.add_argument("--label", required=True, help="nombre de la ejecución")
    run.add_argument("--out", required=True, help="carpeta donde se escribe DIR/NOMBRE")
    run.add_argument("--env", action="append", default=[], metavar="CLAVE=VALOR",
                     help="variable de entorno para el pipeline (se puede repetir)")
    run.add_argument("--extra-arg", action="append", default=[], metavar="ARG",
                     help="argumento adicional para face_identities.py (se puede repetir)")
    run.add_argument("videos", nargs="+", metavar="VIDEO", help="videos a procesar")
    run.set_defaults(handler=command_run)
    compare = sub.add_parser("compare", help="compara dos results.json")
    compare.add_argument("baseline", metavar="BASE.json", help="results.json de la base")
    compare.add_argument("candidate", metavar="CANDIDATO.json", help="results.json del candidato")
    compare.add_argument("--max-recall-drop", type=float, default=DEFAULT_MAX_RECALL_DROP,
                         help="caída máxima de la exhaustividad total, en puntos porcentuales")
    compare.set_defaults(handler=command_compare)
    return parser


def main():
    args = build_parser().parse_args()
    sys.exit(args.handler(args))


if __name__ == "__main__":
    main()
