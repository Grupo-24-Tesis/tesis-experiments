import csv
import glob
import json
import os
import re

import numpy as np
import pandas as pd
import yaml
from openpyxl import load_workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

from harness import flujo_actual, registry, runlog

WORKBOOK = os.path.join(runlog.EXP_DIR, "tuning_results.xlsx")
GT_ROOT = runlog.GT_ROOT
FINDINGS = os.path.join(runlog.EXP_DIR, "findings.yaml")
DECISIONS = os.path.join(runlog.EXP_DIR, "decisions.yaml")
FONT = "Arial"
ENV_CHECKS = "classification/env_checks.csv"
TESTS = ("PRU-01", "ENV-01", "FLU-01", "FLU-02", "FLU-03")
FINDING_COLUMNS = {"id": "id", "date": "fecha", "test_id": "prueba", "finding": "hallazgo",
                   "evidence": "evidencia", "consequence": "consecuencia"}


def read_csv(path):
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def ground_truth():
    frames_rows, face_rows, id_rows = [], [], []
    per_video = {}
    for gt_dir in sorted(glob.glob(os.path.join(GT_ROOT, "*"))):
        stem = os.path.basename(gt_dir)
        doc = runlog.read_json(os.path.join(gt_dir, "frames.json"))
        if not doc:
            continue
        counts = {}
        for key in sorted(doc["frames"], key=int):
            fr = doc["frames"][key]
            ids = sorted({f["identity"] for f in fr["faces"] if f.get("identity") and not f.get("ignore")})
            frames_rows.append({"video": stem, "frame_idx": int(key), "time_s": fr["time_s"],
                                "reviewed": fr.get("reviewed", False), "reviewed_utc": fr.get("reviewed_utc"),
                                "faces": len(fr["faces"]), "ignored": sum(1 for f in fr["faces"] if f.get("ignore")),
                                "identities": ", ".join(ids)})
            for i, f in enumerate(fr["faces"]):
                x1, y1, x2, y2 = f["box"]
                if f.get("identity") and fr.get("reviewed"):
                    counts[f["identity"]] = counts.get(f["identity"], 0) + 1
                face_rows.append({"video": stem, "frame_idx": int(key), "face": i, "reviewed": fr.get("reviewed", False),
                                  "source": f.get("source"), "identity": f.get("identity"), "pose": f.get("pose"),
                                  "occlusion": f.get("occlusion"), "ignore": f.get("ignore", False),
                                  "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                                  "width_px": round(x2 - x1, 1), "height_px": round(y2 - y1, 1),
                                  "smaller_side_px": round(min(x2 - x1, y2 - y1), 1),
                                  **{f"auto_sources.{name}": score
                                     for name, score in (f.get("auto_sources") or {}).items()}})
        per_video[stem] = counts
    for p in runlog.read_json(runlog.IDENTITIES, {"identities": []})["identities"]:
        row = {"identidad": p["id"], "cámara": p.get("camera", ""), "no_seguro_entre_videos": bool(p.get("uncertain")),
               "notas": p.get("notes", "")}
        for stem, counts in per_video.items():
            row[f"rostros_en_fotogramas_revisados.{stem}"] = counts.get(p["id"], 0)
        row["videos"] = sum(1 for counts in per_video.values() if counts.get(p["id"]))
        id_rows.append(row)
    return pd.DataFrame(frames_rows), pd.DataFrame(face_rows), pd.DataFrame(id_rows)


PLACEHOLDER = re.compile(r"\{([^{}:]+)(?::([^{}]*))?\}")


def fill(text, facts):
    def one(m):
        v = facts.get(m.group(1))
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "s/d"
        return format(v, m.group(2) or "")
    return PLACEHOLDER.sub(one, text)


def value_range(series):
    v = pd.to_numeric(series, errors="coerce").dropna()
    return f"{v.min():.2f} a {v.max():.2f}" if len(v) else "s/d"


def run_metrics(df, test_id, label):
    r = df[(df["test_id"] == test_id) & (df["label"] == label)] if not df.empty else df
    if not len(r):
        return None, {}
    return r["run_dir"].iloc[0], runlog.read_json(os.path.join(runlog.EXP_DIR, r["run_dir"].iloc[0], "metrics.json"), {})


def num(x):
    return f"{x:.0f}" if float(x).is_integer() else f"{x:.2f}"


def spec_facts(df, specs):
    facts, used = {}, set()
    for sp in specs or []:
        if "gt_people_of" in sp:
            doc = runlog.read_json(os.path.join(GT_ROOT, sp["gt_people_of"], "frames.json"), {"frames": {}})
            facts[sp["name"]] = len({f["identity"] for fr in doc["frames"].values() if fr.get("reviewed")
                                     for f in fr["faces"] if f.get("identity") and not f.get("ignore")})
            continue
        if "label" in sp:
            run_dir, m = run_metrics(df, sp["test"], sp["label"])
            if run_dir:
                used.add(run_dir)
            facts.update({f"{sp['name']}.{k}": v for k, v in m.items() if v is not None and not k.startswith("_")})
            continue
        sub = df[(df["test_id"] == sp["test"]) & df["label"].astype(str).str.contains(sp["label_contains"], regex=False)]
        v = pd.to_numeric(sub.get(f"metric.{sp['metric']}", pd.Series(dtype=float)), errors="coerce").dropna()
        used.update(sub["run_dir"])
        if not len(v):
            continue
        agg = sp.get("agg", "mean")
        facts[sp["name"]] = ((num(v.min()) if v.min() == v.max() else f"{num(v.min())} a {num(v.max())}") if agg == "range" else
                             float(v.sum()) if agg == "sum" else float(v.mean()) if agg == "mean" else float(v.iloc[0]))
        facts[sp["name"] + ".n_runs"] = int(len(v))
    return facts, used


def decision_facts(df, dec):
    facts = {}
    if "evidence_test" not in dec:
        facts, used = spec_facts(df, dec.get("facts"))
        facts["_used_runs"] = len(used)
        return facts
    rows = df[df["test_id"] == dec["evidence_test"]] if not df.empty else df
    ev = rows[rows["label"] == dec["evidence_label"]] if len(rows) else rows
    if len(ev):
        m = runlog.read_json(os.path.join(runlog.EXP_DIR, ev["run_dir"].iloc[0], "metrics.json"), {})
        facts.update({k: v for k, v in m.items() if v is not None and not k.startswith("_")})
        facts["evidence_run"] = ev["run_dir"].iloc[0]
        for k, v in m.items():
            if k.startswith("xv.within.auc."):
                facts["within_day1" if "Camera_ch6" in k else "within_other"] = v
            if k.startswith("xv.resolution."):
                facts["res_day1" if "Camera_ch6" in k else "res_other"] = v
    if len(rows):
        facts["range.face_auc"] = value_range(rows.get("metric.xv.face.auc", pd.Series(dtype=float)))
        facts["range.person_auc"] = value_range(rows.get("metric.xv.person.auc", pd.Series(dtype=float)))
    emb = df[(df["test_id"] == "EMB-01") & (df.get("param.embed.model") == "Facenet")] if not df.empty else df
    if len(emb) and "metric.emb.auc" in emb:
        facts["emb01_facenet_auc"] = float(emb["metric.emb.auc"].iloc[0])
    cam3 = df[df["video"].astype(str).str.contains("Camera3") & df["test_id"].isin(["E2E-02"])] if not df.empty else df
    if len(cam3):
        v = runlog.read_json(os.path.join(runlog.EXP_DIR, cam3["run_dir"].iloc[0], "run.json"), {}).get("video", {})
        facts["res_cam3"] = f"{v.get('width')}x{v.get('height')} @ {round(v.get('fps') or 0, 1)} fps"
    return facts


def decisions(df):
    docs = yaml.safe_load(open(DECISIONS)) if os.path.exists(DECISIONS) else []
    rows = []
    for dec in docs or []:
        facts = decision_facts(df, dec)
        source = (dec["numbers_from"] + f" ({facts.get('_used_runs', 0)} ejecuciones)" if "numbers_from" in dec else
                  f"Ejecución {facts.get('evidence_run', 's/d')} de {dec['evidence_test']} ({dec['evidence_label']}); hojas T {dec['evidence_test']}, Detalle - xv tamaño y pose")
        head = [("Identificador", dec["id"]), ("Título", dec["title"]), ("Fecha", dec["date"]),
                ("Estado", dec["status"]), ("Decidido por", dec["decided_by"])]
        for item, text in head:
            rows.append({"Decisión": dec["id"], "Sección": "Resumen", "Punto": item, "Texto": text, "Origen de las cifras": ""})
        for sec in dec["sections"]:
            for it in sec["items"]:
                text = fill(it["text"], facts)
                rows.append({"Decisión": dec["id"], "Sección": sec["section"], "Punto": it["item"], "Texto": text,
                             "Origen de las cifras": source if "{" in it["text"] else ""})
    return pd.DataFrame(rows)


def environment():
    rows = []
    runs = sorted(d for d in glob.glob(os.path.join(runlog.RUNS_DIR, "ENV-01", "*"))
                  if runlog.read_json(os.path.join(d, "run.json"), {}).get("status") == "ok")
    if runs:
        run_dir = runs[-1]
        meta = runlog.read_json(os.path.join(run_dir, "run.json"))
        rows.append({"item": "ejecución ENV-01", "value": runlog.rel_exp(run_dir)})
        rows.append({"item": "inicio (UTC)", "value": meta.get("started_utc")})
        git = registry.git_columns(meta.get("git", {}))
        rows.append({"item": "commit del pipeline", "value": git["git_pipeline_commit"]})
        rows.append({"item": "commit de experimentos", "value": git["git_experiments_commit"]})
        for part, device in meta.get("devices_used", {}).items():
            rows.append({"item": f"dispositivo.{part}", "value": device})
        for venv in runlog.read_json(os.path.join(run_dir, "config.json"), {}).get("env", {}).get("venvs", []):
            r = runlog.read_json(os.path.join(run_dir, "outputs", f"env_{venv}.json"), {})
            for k in ("python", "gpu", "driver", "gpu_memory"):
                if k in r:
                    rows.append({"item": f"{venv}.{k}", "value": r[k]})
            for k, v in (r.get("packages") or {}).items():
                rows.append({"item": f"{venv}.package.{k}", "value": v})
            for k, v in (r.get("probe") or {}).items():
                rows.append({"item": f"{venv}.{k}", "value": env_value(k, v)})
    return pd.DataFrame(rows).rename(columns={"item": "elemento", "value": "valor"})


def plain_text(value):
    if isinstance(value, dict):
        return ", ".join(f"{k}: {plain_text(v)}" for k, v in value.items())
    if isinstance(value, (list, tuple)):
        return ", ".join(str(plain_text(v)) for v in value)
    return value


def device_text(value):
    text = plain_text(value)
    if not isinstance(text, str):
        return text
    if "CUDAExecutionProvider" in text:
        return "GPU (CUDA), con CPU de respaldo" if "CPUExecutionProvider" in text else "GPU (CUDA)"
    if text.startswith("cuda"):
        return f"GPU ({text})"
    if "GPU:" in text:
        return f"GPU {text.rsplit(':', 1)[-1]}"
    return text


DEVICE_ITEMS = ("onnx_providers", "tf_gpus", "yolo_device")
DEVICE_CHECKS = ("GPU ",)


def env_value(key, value):
    return device_text(value) if key in DEVICE_ITEMS else plain_text(value)


def env_detail(env_run):
    if env_run.empty:
        return pd.DataFrame()
    rows = read_csv(os.path.join(runlog.EXP_DIR, env_run.iloc[0]["run_dir"], ENV_CHECKS))
    out = []
    for r in rows:
        device_check = r["check"].startswith(DEVICE_CHECKS)
        out.append({"comprobación": r["check"],
                    "entorno": r["venv"] or "todos",
                    "obtenido": device_text(r["value"]) if device_check else r["value"],
                    "esperado": "GPU" if device_check else r["expected"],
                    "resultado": "aprobada" if r["passed"] == "True" else "no aprobada"})
    return pd.DataFrame(out)


def shown_text(value):
    if value is None:
        return ""
    if isinstance(value, float):
        return format(value, ".4f")
    return str(value)


def readme(versions):
    lines = [
        ("Pipeline de reconocimiento facial: resultados de las pruebas del flujo actual", ""),
        ("Generado (UTC)", runlog.utc_now()),
        ("Pipeline medido", versions.get("pipeline", "")),
        ("Cambios sin commit en el pipeline", versions.get("pipeline_cambios", "")),
        ("Experimentos", versions.get("experimentos", "")),
        ("Origen", "runs/flujo_actual/: versiones.txt, estado.txt (código de salida y horas de cada paso), "
                   "un registro .log por paso, compuerta_defecto/ y sin_compuerta/ (results.json y una carpeta "
                   "por video) y entre_camaras/ (extraccion.json por video y resultados.json)"),
        ("Se regenera con", "bench/flujo_actual.sh y después venvs/eval/bin/python -m harness.export_excel"),
        ("Valores", "Todos los valores los mide o calcula el banco de pruebas; nada se escribe a mano. Las "
                    "métricas son las de bench/metrics.py, bench/bench_compuerta.py y bench/entre_camaras.py. "
                    "El único texto escrito a mano es la redacción de Hallazgos (findings.yaml) y de "
                    "Decisiones (decisions.yaml)."),
        ("Pruebas anteriores", "Las pruebas TRK, DET, SEL, EMB, CLU, E2E, XV, ENV-02 y ENV-03 medían el flujo "
                               "por pistas (tracking.py y face_recognition.py), que producción ya no ejecuta. "
                               "Sus ejecuciones siguen en results/registry.csv y runs/<prueba>/ y ya no están en "
                               "este libro; las cifras de Decisiones D001 a D004 salen de ellas."),
        ("Alcance", "Corrida reducida para durar menos de una hora: FLU-01 en 2 videos residenciales, 7 de "
                    "ChokePoint y los 5 clips borde de la compuerta; FLU-02 en Camera_ch6, P1E_S1_C1, "
                    "P2E_S1_C1.1 y los 5 clips borde; FLU-03 con los 7 videos de ChokePoint de FLU-01. La lista "
                    "está en bench/flujo_actual.sh."),
        ("Privacidad", "Este libro no contiene imágenes ni embeddings."),
        ("", ""),
        ("Hoja", "Contenido"),
        ("Resumen", "Una fila por prueba: qué mide, para qué, datos, métrica principal, valor, otras cifras, "
                    "código de salida y horas"),
        ("Flujo - totales", "FLU-01 y FLU-02 por conjunto de videos (residencial, ChokePoint, clips de la "
                            "compuerta y todos): tiempo, detección e identidades con y sin compuerta"),
        ("Flujo - videos", "FLU-01 y FLU-02 por video"),
        ("Entre cámaras", "FLU-03: métricas de búsqueda con el embedding medio de cada detección, por caso"),
        ("Entre cámaras - umbrales", "FLU-03: precisión, exhaustividad y resultados por consulta para cada "
                                     "umbral de similitud"),
        ("Entorno", "Máquina, GPU, versiones de paquetes y dispositivos de la última ejecución de ENV-01"),
        ("Detalle - entorno", "Cada comprobación de esa ejecución de ENV-01"),
        ("Decisiones", "Decisiones tomadas a partir de los resultados, su motivo y cómo revisarlas"),
        ("Hallazgos", "Registro de hallazgos (findings.yaml)"),
        ("GT - *", "Etiquetas manuales de los videos residenciales: fotogramas, rostros y personas"),
        ("", ""),
        ("Prefijo de columna", "Significado"),
        ("compuerta.", "Flujo de producción: presence_gate.py y después face_identities.py solo en los tramos activos"),
        ("base.", "El mismo flujo sin compuerta: face_identities.py sobre todo el video"),
        ("comparacion.", "Compuerta frente a base. detecciones_base_ausentes = detecciones de la base sin una "
                         "detección con compuerta en el mismo fotograma; exhaustividad_compuerta_sobre_base = "
                         "exhaustividad con compuerta medida sobre los fotogramas analizados de la base (solo "
                         "ChokePoint); tiempo_relativo = tiempo con compuerta / tiempo de la base"),
        ("factor", "Segundos de proceso por segundo de video"),
        ("precision, exhaustividad, f1, tp, fp, fn", "Detección de rostros frente a la verdad de terreno"),
        ("etiquetadas, encontradas, duplicados, mezcladas, pureza", "Identidades frente a la verdad de terreno: "
                                                                    "personas etiquetadas, personas con al menos "
                                                                    "una identidad propia, identidades de más, "
                                                                    "identidades con varias personas y fracción "
                                                                    "de rostros asignados a su persona"),
        ("acierto_k, precision_10, rango_medio, ap_media", "Búsqueda entre cámaras: la misma persona entre los k "
                                                           "primeros, fracción correcta de los 10 primeros, "
                                                           "posición media del primer acierto y precisión media"),
        ("azar.", "El mismo valor esperado para un orden al azar"),
        ("", ""),
        ("Prueba", "Qué mide"),
        ("PRU-01", "Pruebas automáticas del banco de pruebas y del worker"),
        ("ENV-01", "El entorno del pipeline funciona"),
        ("FLU-01", "Flujo de producción con los parámetros vigentes"),
        ("FLU-02", "Compuerta de presencia frente al flujo sin compuerta"),
        ("FLU-03", "Búsqueda de la misma persona entre cámaras"),
    ]
    return pd.DataFrame(lines, columns=["Elemento", "Descripción"])


def style(path):
    wb = load_workbook(path)
    head_font = Font(name=FONT, bold=True, color="FFFFFF")
    head_fill = PatternFill("solid", fgColor="1F3A5F")
    body = Font(name=FONT, size=10)
    for ws in wb.worksheets:
        for row in ws.iter_rows():
            for c in row:
                c.font = body
                if isinstance(c.value, float):
                    c.number_format = "0.0000"
        if ws.title == "README":
            ws["A1"].font = Font(name=FONT, bold=True, size=14)
            for r in range(1, ws.max_row + 1):
                if ws.cell(r, 1).value in ("Hoja", "Prefijo de columna", "Prueba"):
                    ws.cell(r, 1).font = ws.cell(r, 2).font = Font(name=FONT, bold=True)
            ws.column_dimensions["A"].width = 26
            ws.column_dimensions["B"].width = 120
            continue
        for c in ws[1]:
            c.font, c.fill = head_font, head_fill
            c.alignment = Alignment(wrap_text=True, vertical="top")
        ws.freeze_panes = "B2" if ws.title in ("Flujo - videos", "Flujo - totales") else "A2"
        if ws.max_row > 1:
            ws.auto_filter.ref = ws.dimensions
        for i, col in enumerate(ws.iter_cols(min_row=1, max_row=ws.max_row), start=1):
            width = max(len(shown_text(c.value)) for c in col)
            ws.column_dimensions[get_column_letter(i)].width = max(8, min(width + 2, 48))
        if ws.title == "Decisiones":
            for letter, width in (("A", 10), ("B", 30), ("C", 34), ("D", 110), ("E", 50)):
                ws.column_dimensions[letter].width = width
            for row in ws.iter_rows(min_row=2):
                for c in row:
                    c.alignment = Alignment(wrap_text=True, vertical="top")
    wb.save(path)


def latest_env_run(df):
    if df.empty or "test_id" not in df.columns:
        return pd.DataFrame()
    env = df[(df["test_id"] == "ENV-01") & (df["status"] == "ok")]
    return env.sort_values("started_utc").tail(1)


def export(run_dir=flujo_actual.DEFAULT_DIR):
    reg = registry.rebuild()
    df = pd.read_csv(reg, keep_default_na=False, na_values=[""]) if os.path.getsize(reg) > 1 else pd.DataFrame()
    env_run = latest_env_run(df)
    env_row = env_run.iloc[0].to_dict() if not env_run.empty else {}
    _, versions = flujo_actual.chain_state(run_dir)
    gated, base, comparisons = flujo_actual.load_runs(run_dir)
    videos = flujo_actual.video_rows(gated, base, comparisons)
    totals = flujo_actual.total_rows(gated, base, comparisons)
    cross_doc = flujo_actual.load_json(os.path.join(run_dir, flujo_actual.CROSS_RESULTS))
    cross = flujo_actual.cross_camera_rows(cross_doc)
    thresholds = flujo_actual.threshold_rows(cross_doc)
    extraction = flujo_actual.extraction_info(cross_doc)
    gt_frames, gt_faces, gt_ids = ground_truth()
    findings = yaml.safe_load(open(FINDINGS)) if os.path.exists(FINDINGS) else []
    sheets = {
        "README": readme(versions),
        "Resumen": flujo_actual.summary_rows(run_dir, env_row, totals, videos, cross, thresholds,
                                             extraction),
        "Flujo - totales": totals,
        "Flujo - videos": videos,
        "Entre cámaras": cross,
        "Entre cámaras - umbrales": thresholds,
        "Entorno": environment(),
        "Detalle - entorno": env_detail(env_run),
        "Decisiones": decisions(df),
        "Hallazgos": pd.DataFrame(findings or []).rename(columns=FINDING_COLUMNS),
        "GT - fotogramas": gt_frames,
        "GT - rostros": gt_faces,
        "GT - personas": gt_ids,
    }
    tmp = WORKBOOK + ".tmp.xlsx"
    with pd.ExcelWriter(tmp, engine="openpyxl") as xw:
        for name, frame in sheets.items():
            (frame if not frame.empty else pd.DataFrame({"(vacío)": ["todavía no hay datos"]})).to_excel(
                xw, sheet_name=name[:31], index=False)
    style(tmp)
    os.replace(tmp, WORKBOOK)
    return WORKBOOK, {k: len(v) for k, v in sheets.items()}


def main():
    path, counts = export()
    print(runlog.rel_exp(path))
    for k, v in counts.items():
        print(f"  {k:<26} {v} filas")


if __name__ == "__main__":
    main()
