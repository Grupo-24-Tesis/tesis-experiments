import argparse
import itertools
import os
import re
import shutil
import sys
from statistics import mean, median

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))

import yaml

from bench import bench_compuerta, mot, pistas
from bench.barrido import parse_name_list, parse_number_list
from harness import runlog
from harness.run import PY, run_logged

TRACKER_YAML = os.path.join(runlog.PIPELINE_DIR, "custom_tracktrack.yaml")
CONFIG_NAME = "config.json"
CANDIDATES_NAME = "candidates"
FACES_NAME = "faces"
MANIFEST_NAME = "manifest.json"
STAGE1_SUMMARY_NAME = "stage1_summary.json"
STAGE2_SUMMARY_NAME = "stage2_summary.json"
STAGES_NAME = "etapas.json"
TRACEBACK_MARK = "Traceback (most recent call last)"
FACES_RESULTS_NAME = "results.json"
TIMES_NAME = "tiempos.json"
MOT_GT_NAME = "gt.txt"
CHECKS_SUFFIX = "_revisiones"
TRACKING_KEY = "seguimiento_s"
STAGE1_KEY = "stage1"
STAGE2_KEY = "stage2"
FACES_KEY = "rostros_s"
FACES_MODE = "rostros"
TRACKS_MODE = "pistas"
SETTING_PATTERN = re.compile(r"^i(\d+)_d(\d+(?:p\d+)?)_m(\d+)$")
BEST_MODE_NOTE = ("El mejor modo de un video es el que tiene más correctos (encontradas - "
                  "duplicados - mezcladas). En un empate gana rostros, el modo actual; entre "
                  "ajustes de pistas empatados gana el primero por nombre. Sin base de rostros "
                  "para un video no se elige mejor modo.")
TRACKING_HEADERS = ["ajuste", "video", "etiquetadas", "pistas", "cubiertas", "duplicadas",
                    "mezcladas", "sin asignar", "pureza", "seguimiento"]
IDENTITY_HEADERS = ["video", "modo", "ajuste", "etiquetadas", "identidades", "encontradas",
                    "duplicados", "mezcladas", "sin asignar", "pureza", "correctos",
                    "cara correcta", "sin verificar", "mejor", "regla"]
OMITTED_HEADERS = ["modo", "origen", "video", "motivo"]
FACES_RULE = "rostros: votos de cada rostro guardado"
RULES_NOTE = ("Reglas de puntuación. Modo rostros (rostros: votos de cada rostro guardado): cada "
              "rostro de una identidad vota por la persona etiquetada que muestra. Modo pistas "
              "(pistas: votos de las cajas de persona): cada caja de persona guardada de una pista "
              "vota por la persona etiquetada que muestra, y la persona principal es la mayoría de "
              "esos votos. En MOT, donde todo fotograma está etiquetado, la pista necesita votos "
              "de al menos la mitad de sus cajas; en ChokePoint y CCTV basta un voto, igual que "
              "en el modo rostros, porque muchos fotogramas no tienen etiqueta. Cara correcta cuenta los rostros guardados que muestran a la persona "
              "principal de su pista según la regla de rostros del conjunto; sin verificar cuenta "
              "los rostros guardados en fotogramas sin etiqueta con que compararlos.")
FOOTAGE_HEADERS = ["video", "altura px", "altura relativa", "personas por revisión",
                   "con personas", "resolución", "mejor modo"]
TRACKING_COUNTS = ["personas_etiquetadas", "pistas", "personas_cubiertas", "pistas_duplicadas",
                   "pistas_mezcladas", "pistas_sin_asignar"]
IDENTITY_COUNTS = ["personas_etiquetadas", "identidades", "encontradas", "duplicados",
                   "mezcladas", "sin_asignar"]
MISSING = "n/d"


def interval_text(value):
    return f"{value:g}".replace(".", "p")


def setting_name(imgsz, interval, min_track):
    return f"i{imgsz}_d{interval_text(interval)}_m{min_track}"


def parse_setting(name):
    match = SETTING_PATTERN.match(name)
    if not match:
        return None
    return {"imgsz": int(match.group(1)), "intervalo": float(match.group(2).replace("p", ".")),
            "min_track": int(match.group(3))}


def settings_grid(imgszs, intervals, min_tracks):
    return [(setting_name(imgsz, interval, min_track), imgsz, interval, min_track)
            for imgsz, interval, min_track in itertools.product(imgszs, intervals, min_tracks)]


def video_stem(path):
    return bench_compuerta.video_name(path)


def unique_videos(paths):
    videos = [os.path.abspath(path) for path in paths]
    seen = {}
    for video in videos:
        stem = video_stem(video)
        if stem in seen:
            raise SystemExit(f"Dos videos tienen el mismo nombre {stem}: {seen[stem]} y {video}")
        seen[stem] = video
    return videos


def run_dir(out, setting, video):
    return os.path.join(out, setting, video_stem(video))


def load_tracker(path=TRACKER_YAML):
    with open(path) as f:
        return yaml.safe_load(f)


def build_config(video, imgsz, interval, min_track, tracker):
    return {"video": os.path.abspath(video),
            "stage1": {"YOLO_IMGSZ": imgsz, "DETECT_INTERVAL_S": interval},
            "tracker": dict(tracker, min_track_len=min_track),
            "stage2": {}}


def candidates_dir(directory):
    return os.path.join(directory, CANDIDATES_NAME)


def faces_dir(directory):
    return os.path.join(directory, FACES_NAME)


def stage1_done(directory):
    return all(os.path.exists(os.path.join(candidates_dir(directory), name))
               for name in (MANIFEST_NAME, STAGE1_SUMMARY_NAME))


def stage2_done(directory):
    return os.path.exists(os.path.join(faces_dir(directory), FACES_RESULTS_NAME))


def stage1_command(directory):
    return [PY["track"], "-m", "harness.stage1_runner", "--config",
            os.path.join(directory, CONFIG_NAME), "--run-dir", directory,
            "--out", candidates_dir(directory)]


def stage2_command(directory):
    return [PY["faces"], "-m", "harness.stage2_runner", "--config",
            os.path.join(directory, CONFIG_NAME), "--run-dir", directory,
            "--candidates", candidates_dir(directory), "--out", faces_dir(directory)]


def statistics_command(video, out):
    return [PY["track"], "-m", "bench.estadisticas_runner", "--video", os.path.abspath(video),
            "--out", out]


def run_command(command, log_prefix):
    return run_logged(command, log_prefix, runlog.isolated_env())


def prepare_stage1(directory, config):
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory)
    runlog.write_json(os.path.join(directory, CONFIG_NAME), config)


def prepare_stage2(directory):
    if os.path.exists(faces_dir(directory)):
        shutil.rmtree(faces_dir(directory))


def read_times(out):
    return runlog.read_json(os.path.join(out, TIMES_NAME), {})


def write_times(out, times):
    runlog.write_json(os.path.join(out, TIMES_NAME), times)


def record_time(times, setting, stem, key, seconds):
    times.setdefault(setting, {}).setdefault(stem, {})[key] = seconds
    return times


def record_stage_code(directory, stage, code):
    path = os.path.join(directory, STAGES_NAME)
    codes = runlog.read_json(path, {})
    codes[stage] = code
    runlog.write_json(path, codes)


def stage_failure(prefix, stage, code, log_path):
    print(f"{prefix}: {stage} falló con código {code}, ver {log_path}", flush=True)


def command_seguimiento(args):
    imgszs = parse_number_list(args.imgsz, int, "--imgsz")
    intervals = parse_number_list(args.intervalo, float, "--intervalo")
    min_tracks = parse_number_list(args.min_track, int, "--min-track")
    videos = unique_videos(args.videos)
    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)
    tracker = load_tracker()
    plan = [(setting, video) for setting in settings_grid(imgszs, intervals, min_tracks)
            for video in videos]
    times = read_times(out)
    failures = 0
    for number, ((name, imgsz, interval, min_track), video) in enumerate(plan, 1):
        stem = video_stem(video)
        directory = run_dir(out, name, video)
        prefix = f"[{number}/{len(plan)}] {name} {stem}"
        if not stage1_problems(directory):
            print(f"{prefix}: seguimiento ya hecho, se omite", flush=True)
            continue
        prepare_stage1(directory, build_config(video, imgsz, interval, min_track, tracker))
        code, seconds = run_command(stage1_command(directory), os.path.join(directory, "stage1"))
        record_stage_code(directory, STAGE1_KEY, code)
        if code != 0 or stage1_problems(directory):
            failures += 1
            stage_failure(prefix, "el seguimiento", code,
                          os.path.join(directory, "stage1.stderr.log"))
            continue
        write_times(out, record_time(times, name, stem, TRACKING_KEY, seconds))
        summary = runlog.read_json(os.path.join(candidates_dir(directory), STAGE1_SUMMARY_NAME))
        print(f"{prefix}: seguimiento en {seconds:.1f} s, {summary['n_tracks']} pistas, "
              f"{summary['n_crops']} recortes", flush=True)
    return 1 if failures else 0


def checked_settings(text):
    names = parse_name_list(text) or []
    if not names:
        raise SystemExit("--ajuste no puede estar vacío")
    for name in names:
        if parse_setting(name) is None:
            raise SystemExit(f"--ajuste espera nombres como i640_d0p5_m3, no {name!r}")
    return names


def command_rostros(args):
    names = checked_settings(args.ajuste)
    videos = unique_videos(args.videos)
    out = os.path.abspath(args.out)
    plan = [(name, video) for name in names for video in videos]
    times = read_times(out)
    failures = 0
    for number, (name, video) in enumerate(plan, 1):
        stem = video_stem(video)
        directory = run_dir(out, name, video)
        prefix = f"[{number}/{len(plan)}] {name} {stem}"
        if not stage1_done(directory):
            failures += 1
            print(f"{prefix}: falta el seguimiento de este ajuste, corra antes seguimiento",
                  flush=True)
            continue
        problems = stage1_problems(directory)
        if problems:
            failures += 1
            print(f"{prefix}: el seguimiento de este ajuste no terminó bien "
                  f"({'; '.join(problems)}), corra antes seguimiento", flush=True)
            continue
        if not stage2_problems(directory):
            print(f"{prefix}: rostros ya hecho, se omite", flush=True)
            continue
        prepare_stage2(directory)
        code, seconds = run_command(stage2_command(directory), os.path.join(directory, "stage2"))
        record_stage_code(directory, STAGE2_KEY, code)
        if code != 0 or stage2_problems(directory):
            failures += 1
            stage_failure(prefix, "rostros", code, os.path.join(directory, "stage2.stderr.log"))
            continue
        write_times(out, record_time(times, name, stem, FACES_KEY, seconds))
        results = runlog.read_json(os.path.join(faces_dir(directory), FACES_RESULTS_NAME))
        print(f"{prefix}: rostros en {seconds:.1f} s, {len(results['faces'])} rostros de "
              f"{results.get('tracks_checked')} pistas", flush=True)
    return 1 if failures else 0


def footage_from_checks(raw):
    checks = raw["checks"]
    heights = [height for _, frame_heights in checks for height in frame_heights]
    counts = [len(frame_heights) for _, frame_heights in checks]
    median_height = median(heights) if heights else 0.0
    return {
        "altura_mediana_px": median_height,
        "altura_relativa": median_height / raw["height"] if raw["height"] else 0.0,
        "personas_por_revision_mediana": median(counts) if counts else 0.0,
        "fraccion_con_personas": sum(1 for c in counts if c) / len(counts) if counts else 0.0,
        "ancho": raw["width"],
        "alto": raw["height"],
        "revisiones": len(checks),
    }


def checks_dir(out_file):
    return os.path.splitext(os.path.abspath(out_file))[0] + CHECKS_SUFFIX


def command_estadisticas(args):
    videos = unique_videos(args.videos)
    out_file = os.path.abspath(args.out)
    raw_dir = checks_dir(out_file)
    os.makedirs(raw_dir, exist_ok=True)
    document = runlog.read_json(out_file, {})
    failures = 0
    for video in videos:
        stem = video_stem(video)
        raw_path = os.path.join(raw_dir, f"{stem}.json")
        if os.path.exists(raw_path):
            print(f"[{stem}] revisiones ya hechas, se reutilizan", flush=True)
        else:
            code, seconds = run_command(statistics_command(video, raw_path),
                                        os.path.join(raw_dir, stem))
            if code != 0 or not os.path.exists(raw_path):
                failures += 1
                stage_failure(f"[{stem}]", "la pasada de personas", code,
                              os.path.join(raw_dir, f"{stem}.stderr.log"))
                continue
            print(f"[{stem}] pasada de personas en {seconds:.1f} s", flush=True)
        stats = dict(footage_from_checks(runlog.read_json(raw_path)), video=video)
        document[stem] = stats
        print(f"[{stem}] altura mediana {stats['altura_mediana_px']:.0f} px "
              f"({stats['altura_relativa']:.1%} del alto), "
              f"{stats['personas_por_revision_mediana']:g} personas por revisión, "
              f"{stats['fraccion_con_personas']:.1%} de revisiones con personas, "
              f"{stats['ancho']}x{stats['alto']}", flush=True)
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    runlog.write_json(out_file, document)
    print(f"Estadísticas escritas en {out_file}")
    return 1 if failures else 0


def spanish_identity(result):
    return {
        "personas_etiquetadas": result["labelled"],
        "identidades": result["identities"],
        "encontradas": result["found"],
        "duplicados": result["duplicates"],
        "mezcladas": result["mixed"],
        "sin_asignar": result["without_person"],
        "pureza": result["purity"],
        "correctos": result["found"] - result["duplicates"] - result["mixed"],
    }


def tracks_identity(stem, manifest, identities, mot_gt,
                    find_gt=bench_compuerta.find_ground_truth):
    if mot_gt is not None:
        result = pistas.mot_identity(manifest, identities, mot_gt)
    else:
        kind, gt, frame_map = find_gt(stem)
        if kind == bench_compuerta.KIND_CHOKEPOINT:
            result = pistas.chokepoint_identity(manifest, identities, gt, frame_map)
        elif kind == bench_compuerta.KIND_CCTV:
            result = pistas.cctv_identity(manifest, identities, manifest["fps"], gt)
        else:
            return None
    return dict(result, regla=pistas.VOTE_RULE)


def has_traceback(path):
    if not os.path.exists(path):
        return False
    with open(path, errors="replace") as f:
        return TRACEBACK_MARK in f.read()


def stage_failures(directory, stage, label):
    reasons = []
    code = runlog.read_json(os.path.join(directory, STAGES_NAME), {}).get(stage)
    if code is not None and code != 0:
        reasons.append(f"la etapa de {label} terminó con código {code}")
    log = f"{stage}.stderr.log"
    if has_traceback(os.path.join(directory, log)):
        reasons.append(f"la etapa de {label} dejó un error (Traceback) en {log}")
    return reasons


def stage1_problems(directory):
    reasons = stage_failures(directory, STAGE1_KEY, "seguimiento")
    if not stage1_done(directory):
        reasons.append(f"sin etapa 1 completa (falta {CANDIDATES_NAME}/{MANIFEST_NAME} o "
                       f"{CANDIDATES_NAME}/{STAGE1_SUMMARY_NAME})")
    return reasons


def stage2_problems(directory):
    reasons = stage_failures(directory, STAGE2_KEY, "rostros")
    if not stage2_done(directory):
        reasons.append(f"sin {FACES_NAME}/{FACES_RESULTS_NAME}")
    elif not os.path.exists(os.path.join(faces_dir(directory), STAGE2_SUMMARY_NAME)):
        reasons.append(f"la etapa de rostros no terminó (falta "
                       f"{FACES_NAME}/{STAGE2_SUMMARY_NAME})")
    return reasons


def omission(mode, origin, stem, reason):
    return {"modo": mode, "origen": origin, "video": stem, "motivo": reason}


def score_run(stem, directory, mot_gt, find_gt=bench_compuerta.find_ground_truth):
    manifest = runlog.read_json(os.path.join(candidates_dir(directory), MANIFEST_NAME))
    problems = stage2_problems(directory)
    identity = None
    if not problems:
        results = runlog.read_json(os.path.join(faces_dir(directory), FACES_RESULTS_NAME))
        identity = tracks_identity(stem, manifest, mot.tracks_to_identities(manifest, results),
                                   mot_gt, find_gt)
        if identity is None:
            problems.append("tipo de verdad de terreno desconocido: no hay gt.txt de MOT, ni "
                            "verdad de terreno de ChokePoint o CCTV para este video")
    return ({"seguimiento": mot.tracking_metrics(manifest, mot_gt) if mot_gt is not None else None,
             "identidad": identity, "con_rostros": stage2_done(directory)}, problems)


def discover_runs(corridas):
    runs = {}
    for name in sorted(os.listdir(corridas)):
        if parse_setting(name) is None:
            continue
        setting_dir = os.path.join(corridas, name)
        if not os.path.isdir(setting_dir):
            continue
        for stem in sorted(os.listdir(setting_dir)):
            directory = os.path.join(setting_dir, stem)
            if os.path.isdir(directory):
                runs.setdefault(name, {})[stem] = directory
    return runs


def score_runs(runs, mot_gt_for, find_gt):
    scores = {}
    omitted = []
    for setting, by_video in runs.items():
        for stem, directory in by_video.items():
            problems = stage1_problems(directory)
            if problems:
                omitted.extend(omission(TRACKS_MODE, setting, stem, reason) for reason in problems)
                continue
            score, problems = score_run(stem, directory, mot_gt_for(stem), find_gt)
            omitted.extend(omission(TRACKS_MODE, setting, stem, reason) for reason in problems)
            scores.setdefault(setting, {})[stem] = score
    return scores, omitted


def mot_loader(mot_dir):
    cache = {}

    def load(stem):
        if stem not in cache:
            path = os.path.join(mot_dir, stem, MOT_GT_NAME)
            cache[stem] = mot.load_mot_ground_truth(path) if os.path.exists(path) else None
        return cache[stem]
    return load


def baseline_identities(results_path, mot_gt_for):
    document = runlog.read_json(results_path)
    root = os.path.dirname(os.path.abspath(results_path))
    origin = os.path.abspath(results_path)
    found = {}
    omitted = []
    for name, entry in document["videos"].items():
        if "error" in entry:
            omitted.append(omission(FACES_MODE, origin, name,
                                    f"la base marcó un error: {entry['error']}"))
            continue
        gt = mot_gt_for(name)
        if gt is not None:
            if "detections_file" not in entry:
                omitted.append(omission(FACES_MODE, origin, name,
                                        "la base no indica detections_file para hallar "
                                        "identities.json"))
                continue
            path = os.path.join(root, os.path.dirname(entry["detections_file"]),
                                bench_compuerta.IDENTITIES_NAME)
            found[name] = mot.identity_metrics(runlog.read_json(path)["identities"], gt)
        elif entry.get("identity"):
            found[name] = spanish_identity(entry["identity"])
        elif entry.get("kind") is None:
            omitted.append(omission(FACES_MODE, origin, name,
                                    "tipo de verdad de terreno desconocido: no hay gt.txt de "
                                    "MOT y la base no tiene métricas de identidad"))
        else:
            omitted.append(omission(FACES_MODE, origin, name,
                                    "la base no tiene métricas de identidad"))
    return found, omitted


def merge_baselines(paths, mot_gt_for):
    merged = {}
    sources = {}
    omitted = []
    for path in paths:
        found, skipped = baseline_identities(path, mot_gt_for)
        omitted.extend(skipped)
        for name, identity in found.items():
            if name in merged:
                print(f"Aviso: {name} aparece en {sources[name]} y en {path}; se usa el primero")
                continue
            merged[name] = identity
            sources[name] = path
    return merged, sources, omitted


def best_mode(faces, tracks):
    if faces is None:
        return None
    best = {"modo": FACES_MODE, "ajuste": None, "correctos": faces["correctos"]}
    for setting in sorted(tracks):
        identity = tracks[setting]
        if identity is not None and identity["correctos"] > best["correctos"]:
            best = {"modo": TRACKS_MODE, "ajuste": setting, "correctos": identity["correctos"]}
    return best


def best_text(best):
    if best is None:
        return MISSING
    return best["modo"] if best["ajuste"] is None else f"{best['modo']} {best['ajuste']}"


def fmt_pct(value):
    return MISSING if value is None else f"{value:.1%}"


def fmt_seconds(value):
    return MISSING if value is None else f"{value:.1f} s"


def tracking_total(per_video):
    if not per_video:
        return None
    total = {key: sum(m[key] for m in per_video.values()) for key in TRACKING_COUNTS}
    total["videos"] = len(per_video)
    total["pureza_media"] = mean(m["pureza"] for m in per_video.values())
    return total


def tracking_by_setting(scores):
    return {setting: {stem: run["seguimiento"] for stem, run in runs.items()
                      if run["seguimiento"] is not None}
            for setting, runs in scores.items()}


def tracking_rows(scores, times):
    rows = []
    for setting, per_video in sorted(tracking_by_setting(scores).items()):
        if not per_video:
            continue
        seconds = []
        for stem, m in sorted(per_video.items()):
            elapsed = times.get(setting, {}).get(stem, {}).get(TRACKING_KEY)
            seconds.append(elapsed)
            rows.append([setting, stem] + [str(m[key]) for key in TRACKING_COUNTS]
                        + [fmt_pct(m["pureza"]), fmt_seconds(elapsed)])
        total = tracking_total(per_video)
        total_seconds = None if None in seconds else sum(seconds)
        rows.append([setting, "total"] + [str(total[key]) for key in TRACKING_COUNTS]
                    + [fmt_pct(total["pureza_media"]), fmt_seconds(total_seconds)])
    return rows


def identity_cells(identity, tracks_mode):
    if identity is None:
        return [MISSING] * (len(IDENTITY_COUNTS) + 4)
    face_cells = ([str(identity["cara_correcta"]), str(identity["cara_sin_verificar"])]
                  if tracks_mode else ["-", "-"])
    return ([str(identity[key]) for key in IDENTITY_COUNTS]
            + [fmt_pct(identity["pureza"]), str(identity["correctos"])] + face_cells)


def tracks_identities(scores, stem):
    return {setting: runs[stem]["identidad"] for setting, runs in scores.items()
            if stem in runs and runs[stem]["identidad"] is not None}


def compared_videos(scores):
    return sorted({stem for runs in scores.values() for stem, run in runs.items()
                   if run["identidad"] is not None})


def identity_rows(faces, scores, best):
    rows = []
    for stem in compared_videos(scores):
        choice = best.get(stem)
        mark = "sí" if choice is not None and choice["modo"] == FACES_MODE else ""
        rows.append([stem, FACES_MODE, "-"] + identity_cells(faces.get(stem), False)
                    + [mark, FACES_RULE])
        for setting, identity in sorted(tracks_identities(scores, stem).items()):
            mark = "sí" if choice is not None and choice["ajuste"] == setting else ""
            rows.append([stem, TRACKS_MODE, setting] + identity_cells(identity, True)
                        + [mark, identity.get("regla", pistas.VOTE_RULE)])
    return rows


def omitted_rows(omitted):
    return [[item["modo"], item["origen"], item["video"], item["motivo"]] for item in omitted]


def footage_rows(stats, best):
    rows = []
    for stem, s in sorted(stats.items()):
        rows.append([stem, f"{s['altura_mediana_px']:.0f}", fmt_pct(s["altura_relativa"]),
                     f"{s['personas_por_revision_mediana']:g}", fmt_pct(s["fraccion_con_personas"]),
                     f"{s['ancho']}x{s['alto']}", best_text(best.get(stem))])
    return rows


def format_table(headers, rows):
    widths = [max(len(str(cell)) for cell in column) for column in zip(headers, *rows)]
    lines = ["  ".join(str(cell).ljust(width) for cell, width in zip(line, widths)).rstrip()
             for line in [headers, ["-" * width for width in widths], *rows]]
    return "\n".join(lines)


def print_table(title, headers, rows, empty):
    print(title)
    print(format_table(headers, rows) if rows else f"  {empty}")
    print()


def command_puntuar(args):
    corridas = os.path.abspath(args.corridas)
    mot_gt_for = mot_loader(os.path.abspath(args.mot_gt))
    stats_path = os.path.abspath(args.estadisticas)
    if not os.path.exists(stats_path):
        raise SystemExit(f"No existe el archivo de estadísticas {stats_path}: corra antes "
                         f"estadisticas")
    scores, omitted_runs = score_runs(discover_runs(corridas), mot_gt_for,
                                      bench_compuerta.find_ground_truth)
    faces, sources, omitted_bases = merge_baselines(args.base, mot_gt_for)
    omitted = omitted_runs + omitted_bases
    best = {stem: best_mode(faces.get(stem), tracks_identities(scores, stem))
            for stem in compared_videos(scores)}
    stats = runlog.read_json(stats_path)
    times = read_times(corridas)
    print_table("Seguimiento por ajuste en MOT (etapa 1, sin rostros)", TRACKING_HEADERS,
                tracking_rows(scores, times), "sin videos con verdad de terreno MOT")
    print_table("Identidades por video: modo rostros frente a cada ajuste de pistas con rostros",
                IDENTITY_HEADERS, identity_rows(faces, scores, best),
                "sin ajustes de pistas con rostros puntuables")
    print(RULES_NOTE)
    print()
    print_table("Estadísticas del metraje y mejor modo por video", FOOTAGE_HEADERS,
                footage_rows(stats, best), "sin estadísticas")
    print_table("Omitidos", OMITTED_HEADERS, omitted_rows(omitted), "ninguno")
    print(BEST_MODE_NOTE)
    document = {
        "nota": BEST_MODE_NOTE,
        "reglas": {FACES_MODE: FACES_RULE, TRACKS_MODE: pistas.VOTE_RULE, "nota": RULES_NOTE},
        "corridas": corridas,
        "bases": [os.path.abspath(path) for path in args.base],
        "mot_gt": os.path.abspath(args.mot_gt),
        "estadisticas": stats_path,
        "ajustes": {setting: {"parametros": parse_setting(setting), "videos": runs}
                    for setting, runs in scores.items()},
        "totales_seguimiento": {setting: tracking_total(per_video)
                                for setting, per_video in tracking_by_setting(scores).items()
                                if per_video},
        "rostros": faces,
        "origen_rostros": sources,
        "mejor_modo": best,
        "metraje": stats,
        "tiempos": times,
        "omitidos": omitted,
    }
    out_file = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    runlog.write_json(out_file, document)
    print(f"Resultados escritos en {out_file}")
    return 0


def build_parser():
    parser = argparse.ArgumentParser(
        description="Banco del modo de pistas para cámaras elevadas: corre las etapas 1 y 2 del "
                    "pipeline por ajuste, mide el metraje y compara con el modo de rostros")
    sub = parser.add_subparsers(dest="command", required=True)
    tracking = sub.add_parser("seguimiento", help="corre la etapa 1 para cada combinación de "
                                                  "ajustes y video")
    tracking.add_argument("--out", required=True, help="carpeta de corridas (DIR/AJUSTE/VIDEO)")
    tracking.add_argument("--imgsz", default="640", help="valores de YOLO_IMGSZ separados por comas")
    tracking.add_argument("--intervalo", default="0.5",
                          help="valores de DETECT_INTERVAL_S en segundos separados por comas")
    tracking.add_argument("--min-track", default="3",
                          help="valores de min_track_len del tracker separados por comas")
    tracking.add_argument("videos", nargs="+", metavar="VIDEO", help="videos a procesar")
    tracking.set_defaults(handler=command_seguimiento)
    faces = sub.add_parser("rostros", help="corre la etapa 2 para los ajustes indicados")
    faces.add_argument("--out", required=True, help="carpeta de corridas usada en seguimiento")
    faces.add_argument("--ajuste", required=True,
                       help="ajustes separados por comas, por ejemplo i640_d0p5_m3")
    faces.add_argument("videos", nargs="+", metavar="VIDEO", help="videos a procesar")
    faces.set_defaults(handler=command_rostros)
    stats = sub.add_parser("estadisticas", help="mide personas y altura de sus cajas por video, "
                                                "como las revisiones de la compuerta")
    stats.add_argument("--out", required=True, help="archivo JSON de salida")
    stats.add_argument("videos", nargs="+", metavar="VIDEO", help="videos a procesar")
    stats.set_defaults(handler=command_estadisticas)
    score = sub.add_parser("puntuar", help="puntúa las corridas y las compara con el modo de "
                                           "rostros")
    score.add_argument("--corridas", required=True, help="carpeta de corridas")
    score.add_argument("--base", action="append", required=True, metavar="RESULTS_JSON",
                       help="results.json del modo de rostros (se puede repetir)")
    score.add_argument("--mot-gt", required=True, metavar="MOTDIR",
                       help="carpeta con MOTDIR/VIDEO/gt.txt")
    score.add_argument("--estadisticas", required=True, metavar="FILE",
                       help="archivo JSON escrito por estadisticas")
    score.add_argument("--out", required=True, metavar="FILE.json", help="archivo JSON de salida")
    score.set_defaults(handler=command_puntuar)
    return parser


def main():
    args = build_parser().parse_args()
    sys.exit(args.handler(args))


if __name__ == "__main__":
    main()
