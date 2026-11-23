import json
import os
import re

import pandas as pd

from bench import bench_compuerta, metrics
from harness import runlog

DEFAULT_DIR = os.path.join(runlog.RUNS_DIR, "flujo_actual")
GATED_LABEL = "compuerta_defecto"
BASE_LABEL = "sin_compuerta"
RESULTS_NAME = "results.json"
CROSS_RESULTS = os.path.join("entre_camaras", "resultados.json")
DATASET_RESIDENTIAL = "residencial"
DATASET_CHOKEPOINT = "ChokePoint"
DATASET_CLIPS = "clips de la compuerta"
DATASET_ALL = "todos"
DATASETS = (DATASET_RESIDENTIAL, DATASET_CHOKEPOINT, DATASET_CLIPS)
VARIANTS = {"media": "media de la identidad"}
CASES = {"todas": "todas las demás cámaras",
         "misma_pasada": "otra cámara de la misma pasada",
         "misma_sesion": "otra pasada de la misma sesión",
         "otra_sesion": "otra sesión del mismo portal",
         "otro_portal": "otro portal"}
ADOPTED_THRESHOLD = 0.55
STEPS = {
    "PRU-01": ("pruebas",),
    "ENV-01": ("entorno",),
    "FLU-01": ("compuerta",),
    "FLU-02": ("base", "comparacion"),
    "FLU-03": ("extraccion", "entre_camaras"),
}
PYTEST_COUNT = re.compile(r"(\d+) (passed|failed|error|errors|skipped)")
REGRESSION_PREFIX = "REGRESIÓN:"


def load_json(path, default=None):
    if not os.path.exists(path):
        return default
    with open(path) as f:
        return json.load(f)


def read_lines(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return f.read().splitlines()


def dataset_of(entry):
    video = entry.get("video", "")
    if f"{os.sep}videos_prueba{os.sep}" in video:
        return DATASET_CLIPS
    if f"{os.sep}chokepoint{os.sep}" in video:
        return DATASET_CHOKEPOINT
    return DATASET_RESIDENTIAL


def chain_state(run_dir):
    steps = {}
    for line in read_lines(os.path.join(run_dir, "estado.txt")):
        parts = line.split()
        if len(parts) == 4:
            steps[parts[0]] = {"codigo": int(parts[1]), "inicio": parts[2], "fin": parts[3]}
    versions = {}
    for line in read_lines(os.path.join(run_dir, "versiones.txt")):
        key, _, value = line.partition(" ")
        versions[key] = value
    return steps, versions


def pytest_counts(run_dir):
    counts = {}
    for line in reversed(read_lines(os.path.join(run_dir, "pruebas.log"))):
        found = PYTEST_COUNT.findall(line)
        if found:
            for number, kind in found:
                counts[kind.rstrip("s") if kind == "errors" else kind] = int(number)
            break
    return counts


def regressions(run_dir):
    return [line[len(REGRESSION_PREFIX):].strip()
            for line in read_lines(os.path.join(run_dir, "comparacion.log"))
            if line.startswith(REGRESSION_PREFIX)]


def detections_doc(label_dir, entry):
    return load_json(os.path.join(label_dir, entry["detections_file"]))


def comparable(gated, base):
    return bool(gated and base and "error" not in gated and "error" not in base)


def compare_video(name, gated, base, gated_dir, base_dir):
    gated_doc = detections_doc(gated_dir, gated)
    base_doc = detections_doc(base_dir, base)
    missing = metrics.missing_detections(base_doc["detections"], gated_doc["detections"])
    rescored = bench_compuerta.gated_detection_over_baseline(name, base_doc, gated_doc)
    return {"missing": len(missing), "rescored": rescored}


def load_runs(run_dir):
    gated = load_json(os.path.join(run_dir, GATED_LABEL, RESULTS_NAME), {"videos": {}})
    base = load_json(os.path.join(run_dir, BASE_LABEL, RESULTS_NAME), {"videos": {}})
    comparisons = {}
    for name, entry in gated["videos"].items():
        other = base["videos"].get(name)
        if comparable(entry, other):
            comparisons[name] = compare_video(name, entry, other,
                                              os.path.join(run_dir, GATED_LABEL),
                                              os.path.join(run_dir, BASE_LABEL))
    return gated, base, comparisons


def identity_columns(prefix, identity):
    identity = identity or {}
    return {f"{prefix}etiquetadas": identity.get("labelled"),
            f"{prefix}encontradas": identity.get("found"),
            f"{prefix}duplicados": identity.get("duplicates"),
            f"{prefix}mezcladas": identity.get("mixed"),
            f"{prefix}sin_persona": identity.get("without_person"),
            f"{prefix}pureza": identity.get("purity")}


def detection_columns(prefix, detection):
    detection = detection or {}
    return {f"{prefix}precision": detection.get("precision"),
            f"{prefix}exhaustividad": detection.get("recall"),
            f"{prefix}f1": detection.get("f1"),
            f"{prefix}tp": detection.get("tp"),
            f"{prefix}fp": detection.get("fp"),
            f"{prefix}fn": detection.get("fn")}


def run_columns(prefix, entry):
    return {f"{prefix}tiempo_s": entry.get("wall_s"),
            f"{prefix}factor": entry.get("factor"),
            f"{prefix}fotogramas_analizados": entry.get("analysed_frames"),
            f"{prefix}detecciones": entry.get("detections_count"),
            f"{prefix}identidades": entry.get("identities_count")}


def video_rows(gated, base, comparisons):
    rows = []
    for name, entry in gated["videos"].items():
        other = base["videos"].get(name) or {}
        row = {"video": name, "conjunto": dataset_of(entry),
               "verdad_de_terreno": entry.get("kind"),
               "duracion_s": entry.get("duration_s") or other.get("duration_s"),
               "error": entry.get("error") or other.get("error")}
        row.update(run_columns("compuerta.", entry))
        row.update({"compuerta.tiempo_compuerta_s": entry.get("gate_wall_s"),
                    "compuerta.tiempo_rostros_s": entry.get("face_wall_s"),
                    "compuerta.fraccion_activa": entry.get("active_fraction"),
                    "compuerta.cambios": entry.get("switch_count")})
        row.update(detection_columns("compuerta.", entry.get("detection")))
        row.update(identity_columns("compuerta.", entry.get("identity")))
        row.update(run_columns("base.", other))
        row.update(detection_columns("base.", other.get("detection")))
        row.update(identity_columns("base.", other.get("identity")))
        comparison = comparisons.get(name)
        rescored = (comparison or {}).get("rescored") or {}
        row.update({
            "comparacion.detecciones_base_ausentes": comparison["missing"] if comparison else None,
            "comparacion.exhaustividad_compuerta_sobre_base": rescored.get("recall"),
            "comparacion.tiempo_relativo": (entry["wall_s"] / other["wall_s"]
                                           if comparison and other.get("wall_s") else None),
        })
        rows.append(row)
    return pd.DataFrame(rows, dtype=object)


def effective_gated(gated, comparisons):
    videos = {}
    for name, comparison in comparisons.items():
        entry = gated["videos"][name]
        if comparison["rescored"] is not None:
            entry = {**entry, "detection": comparison["rescored"]}
        videos[name] = entry
    return videos


def sum_of(videos, key):
    return sum(v.get(key) or 0 for v in videos.values())


def factor_of(videos):
    duration = sum_of(videos, "duration_s")
    return sum_of(videos, "wall_s") / duration if duration else None


def labelled_of(videos):
    return sum((v.get("identity") or {}).get("labelled", 0) for v in videos.values())


def total_row(label, gated_videos, base_videos, rescored_videos, missing):
    gated_detection = bench_compuerta.total_detection(gated_videos) or {}
    base_detection = bench_compuerta.total_detection(base_videos) or {}
    rescored_detection = bench_compuerta.total_detection(rescored_videos) or {}
    gated_identity = bench_compuerta.total_identity(gated_videos) or {}
    base_identity = bench_compuerta.total_identity(base_videos) or {}
    row = {"conjunto": label, "videos": len(gated_videos),
           "videos_con_verdad_de_terreno": sum(1 for v in gated_videos.values() if v.get("kind")),
           "duracion_s": sum_of(gated_videos, "duration_s"),
           "compuerta.tiempo_s": sum_of(gated_videos, "wall_s"),
           "compuerta.factor": factor_of(gated_videos),
           "compuerta.fotogramas_analizados": sum_of(gated_videos, "analysed_frames"),
           "compuerta.detecciones": sum_of(gated_videos, "detections_count"),
           "compuerta.identidades": sum_of(gated_videos, "identities_count")}
    row.update(detection_columns("compuerta.", gated_detection))
    row.update({"compuerta.etiquetadas": labelled_of(gated_videos) or None,
                "compuerta.encontradas": gated_identity.get("found"),
                "compuerta.duplicados": gated_identity.get("duplicates"),
                "compuerta.mezcladas": gated_identity.get("mixed"),
                "compuerta.pureza": gated_identity.get("purity"),
                "base.tiempo_s": sum_of(base_videos, "wall_s"),
                "base.factor": factor_of(base_videos),
                "base.fotogramas_analizados": sum_of(base_videos, "analysed_frames"),
                "base.detecciones": sum_of(base_videos, "detections_count"),
                "base.identidades": sum_of(base_videos, "identities_count")})
    row.update(detection_columns("base.", base_detection))
    row.update({"base.encontradas": base_identity.get("found"),
                "base.duplicados": base_identity.get("duplicates"),
                "base.mezcladas": base_identity.get("mixed"),
                "base.pureza": base_identity.get("purity"),
                "comparacion.detecciones_base_ausentes": missing,
                "comparacion.exhaustividad_compuerta_sobre_base": rescored_detection.get("recall"),
                "comparacion.caida_exhaustividad_pp": (
                    bench_compuerta.points(base_detection["recall"], rescored_detection["recall"])
                    if base_detection and rescored_detection else None)})
    return row


def total_rows(gated, base, comparisons):
    rescored_all = effective_gated(gated, comparisons)
    groups = [(label, [n for n in comparisons if dataset_of(gated["videos"][n]) == label])
              for label in DATASETS]
    groups.append((DATASET_ALL, list(comparisons)))
    rows = []
    for label, names in groups:
        if not names:
            continue
        rows.append(total_row(label,
                              {n: gated["videos"][n] for n in names},
                              {n: base["videos"][n] for n in names},
                              {n: rescored_all[n] for n in names},
                              sum(comparisons[n]["missing"] for n in names)))
    return pd.DataFrame(rows, dtype=object)


def reported(document):
    results = (document or {}).get("resultados", {})
    return [(variant, results[variant]) for variant in VARIANTS if variant in results]


def cross_camera_rows(document):
    rows = []
    for variant, cases in reported(document):
        for case, block in cases.items():
            found = block["metricas"]
            row = {"representacion": VARIANTS.get(variant, variant), "variante": variant,
                   "caso": CASES.get(case, case), "clave_caso": case,
                   "consultas": found.get("consultas"), "sin_objetivo": block.get("sin_objetivo")}
            for key, value in found.items():
                if key == "consultas":
                    continue
                if isinstance(value, dict):
                    for chance_key, chance_value in value.items():
                        row[f"azar.{chance_key}"] = chance_value
                else:
                    row[key] = value
            rows.append(row)
    return pd.DataFrame(rows)


def threshold_rows(document):
    rows = []
    for variant, cases in reported(document):
        for case, block in cases.items():
            for entry in block.get("umbrales", []):
                rows.append({"representacion": VARIANTS.get(variant, variant),
                             "caso": CASES.get(case, case), **entry})
    return pd.DataFrame(rows)


def extraction_info(document):
    info = (document or {}).get("extraccion", {})
    return {"videos": info.get("videos"), "identidades": info.get("identidades"),
            "etiquetadas": info.get("etiquetadas"), "sin_persona": info.get("sin_persona"),
            "mezcladas": info.get("mezcladas"),
            "videos_distintos": len(info.get("videos_distintos", [])),
            "videos_mejor_distinto": len(info.get("videos_mejor_distinto", []))}


def pct(value):
    return "-" if value is None else f"{value * 100:.1f} %".replace(".", ",")


def dec(value, digits=2):
    return "-" if value is None else f"{value:.{digits}f}".replace(".", ",")


def lookup(frame, column, key):
    if frame.empty or column not in frame.columns:
        return {}
    match = frame[frame[column] == key]
    return match.iloc[0].to_dict() if not match.empty else {}


def identity_text(row):
    if not row or row.get("compuerta.etiquetadas") is None:
        return "-"
    return (f"{row['compuerta.encontradas']} de {row['compuerta.etiquetadas']} personas encontradas, "
            f"{row['compuerta.duplicados']} duplicados, {row['compuerta.mezcladas']} mezcladas, "
            f"pureza {pct(row['compuerta.pureza'])}")


def detection_text(row):
    if not row or row.get("compuerta.precision") is None:
        return "-"
    return (f"precisión {pct(row['compuerta.precision'])}, exhaustividad "
            f"{pct(row['compuerta.exhaustividad'])}, F1 {pct(row['compuerta.f1'])}")


def summary_rows(run_dir, env_row, totals, videos, cross, thresholds, extraction):
    steps, _ = chain_state(run_dir)
    counts = pytest_counts(run_dir)
    found = regressions(run_dir)
    chokepoint = lookup(totals, "conjunto", DATASET_CHOKEPOINT)
    residential = lookup(totals, "conjunto", DATASET_RESIDENTIAL)
    everything = lookup(totals, "conjunto", DATASET_ALL)
    long_video = lookup(videos, "video", "largo")
    long_text = (f"; video largo {dec(long_video.get('base.factor'))} y "
                 f"{dec(long_video.get('compuerta.factor'))}" if long_video else "")
    datasets_text = ", ".join(f"{int(row['videos'])} {row['conjunto']}"
                              for _, row in totals.iterrows()
                              if row["conjunto"] != DATASET_ALL) if not totals.empty else "-"
    mean_all = {}
    if not cross.empty:
        match = cross[(cross["variante"] == "media") & (cross["clave_caso"] == "todas")]
        mean_all = match.iloc[0].to_dict() if not match.empty else {}
    by_case = {}
    if not cross.empty:
        for _, row in cross[cross["variante"] == "media"].iterrows():
            by_case[row["clave_caso"]] = row["acierto_1"]
    adopted = {}
    if not thresholds.empty:
        match = thresholds[(thresholds["representacion"] == VARIANTS["media"])
                           & (thresholds["caso"] == CASES["todas"])
                           & (thresholds["umbral"].round(2) == ADOPTED_THRESHOLD)]
        adopted = match.iloc[0].to_dict() if not match.empty else {}
    rows = [
        {"Prueba": "PRU-01",
         "Descripción": "Pruebas automáticas del banco de pruebas y del worker (pytest)",
         "Para qué": "Comprobar que las herramientas de medición y el worker se comportan como "
                     "se espera antes de confiar en sus cifras",
         "Datos": "datos sintéticos de las pruebas en tests/",
         "Métrica principal": "pruebas aprobadas",
         "Valor": counts.get("passed"),
         "Otras cifras": f"fallidas {counts.get('failed', 0)}, errores {counts.get('error', 0)}"},
        {"Prueba": "ENV-01",
         "Descripción": "El entorno funciona: librerías, GPU y lectura de video en los venvs track, "
                        "faces, eval y worker",
         "Para qué": "Descartar que un resultado dependa de un entorno roto o mezclado",
         "Datos": "los 3 videos residenciales",
         "Métrica principal": "fracción de comprobaciones aprobadas",
         "Valor": env_row.get("metric.env.pass_rate"),
         "Otras cifras": (f"{int(env_row['metric.env.passed'])} de {int(env_row['metric.env.checks'])} "
                          f"comprobaciones") if env_row else "-"},
        {"Prueba": "FLU-01",
         "Descripción": "Flujo de producción completo (compuerta de presencia y face_identities.py) "
                        "con los parámetros vigentes",
         "Para qué": "Medir cuántos rostros detecta, con qué precisión, y si cada persona queda en "
                     "una sola identidad por video",
         "Datos": f"{int(everything.get('videos') or 0)} videos: {datasets_text}",
         "Métrica principal": "exhaustividad de detección en ChokePoint",
         "Valor": chokepoint.get("compuerta.exhaustividad"),
         "Otras cifras": (f"ChokePoint: {detection_text(chokepoint)}; {identity_text(chokepoint)}. "
                          f"Residencial: {detection_text(residential)}; {identity_text(residential)}")},
        {"Prueba": "FLU-02",
         "Descripción": "Compuerta de presencia frente al mismo flujo sin compuerta",
         "Para qué": "Confirmar que la compuerta no pierde detecciones y medir cuánto tiempo ahorra",
         "Datos": "los mismos videos de FLU-01",
         "Métrica principal": "detecciones de la base ausentes con compuerta",
         "Valor": everything.get("comparacion.detecciones_base_ausentes"),
         "Otras cifras": (f"regresiones {len(found)}; caída de exhaustividad "
                          f"{dec(everything.get('comparacion.caida_exhaustividad_pp'))} pp; factor de "
                          f"tiempo {dec(everything.get('base.factor'))} sin compuerta y "
                          f"{dec(everything.get('compuerta.factor'))} con compuerta{long_text}")},
        {"Prueba": "FLU-03",
         "Descripción": "Búsqueda de la misma persona en otras cámaras por el embedding de la identidad",
         "Para qué": "Medir si el embedding medio por identidad que envía el worker encuentra a la "
                     "misma persona en otros videos",
         "Datos": f"{extraction.get('videos')} videos de ChokePoint de FLU-01, "
                  f"{extraction.get('identidades')} detecciones con el embedding que envía el worker",
         "Métrica principal": "acierto@1 con la media de la identidad, todas las demás cámaras",
         "Valor": mean_all.get("acierto_1"),
         "Otras cifras": (f"azar {dec(mean_all.get('azar.acierto_1'), 3)}; otra sesión "
                          f"{dec(by_case.get('otra_sesion'), 3)}; otro portal "
                          f"{dec(by_case.get('otro_portal'), 3)}; umbral {dec(ADOPTED_THRESHOLD)}: "
                          f"precisión {pct(adopted.get('precision'))}, exhaustividad "
                          f"{pct(adopted.get('exhaustividad'))}")},
    ]
    for row in rows:
        row.update(step_columns(steps, STEPS[row["Prueba"]]))
    return pd.DataFrame(rows, dtype=object)


def step_columns(steps, names):
    found = [steps[name] for name in names if name in steps]
    complete = len(found) == len(names)
    return {"Código de salida": max(s["codigo"] for s in found) if complete else None,
            "Inicio (UTC)": found[0]["inicio"] if complete else None,
            "Fin (UTC)": found[-1]["fin"] if complete else None}
