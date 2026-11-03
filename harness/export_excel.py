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

from harness import registry, runlog

WORKBOOK = os.path.join(registry.RESULTS_DIR, "tuning_results.xlsx")
GT_ROOT = runlog.GT_ROOT
FINDINGS = os.path.join(runlog.EXP_DIR, "findings.yaml")
DECISIONS = os.path.join(runlog.EXP_DIR, "decisions.yaml")
FONT = "Arial"
ITEM_FILES = [
    ("Detalle - detecciones", "classification/det_matches.csv"),
    ("Detalle - curvas PR", "classification/pr_curve.csv"),
    ("Detalle - candidatos", "classification/candidates.csv"),
    ("Detalle - selección", "classification/selection.csv"),
    ("Detalle - clusters", "outputs/clusters.csv"),
    ("Detalle - errores de cluster", "classification/cluster_errors.csv"),
    ("Detalle - pares de identidades", "classification/pairs_summary.csv"),
    ("Detalle - personas e2e", "classification/e2e_people.csv"),
    ("Detalle - personas", "classification/people.csv"),
    ("Detalle - xv coincidencias", "classification/xv_matches.csv"),
    ("Detalle - xv pares de personas", "classification/xv_person_pairs.csv"),
    ("Detalle - xv barrido de umbral", "classification/xv_threshold_sweep.csv"),
    ("Detalle - xv tamaño y pose", "classification/xv_face_slices.csv"),
]
TEST_DESCRIPTIONS = {
    "ENV-01": "El entorno funciona (librerías, dispositivos, lectura de videos)",
    "ENV-02": "Variación entre ejecuciones repetidas de la configuración base (3 repeticiones)",
    "ENV-03": "La refactorización y las pruebas reproducen la salida de producción",
    "DET-01": "Detectores de rostros sobre fotogramas completos",
    "DET-02": "Umbral de confianza (curva PR de cada ejecución de DET-01)",
    "DET-03": "Tamaño mínimo de rostro que se cuenta",
    "DET-04": "Detección tal como la hace el pipeline (recorte de persona y margen)",
    "DET-05": "DET-04 repetida en los videos nuevos (margen; tamaño mínimo de rostro solo como control)",
    "TRK-01": "Muestreo de fotogramas y ajustes del detector de personas",
    "TRK-02": "Ajustes de asociación del tracker",
    "TRK-03": "Recortes guardados por pista; control de cortes de escena",
    "SEL-01": "Regla para elegir el mejor rostro de cada pista",
    "SEL-02": "Cantidad de rostros por pista",
    "EMB-00": "Control sin etiquetas de los embeddings (rostros frente a fondo)",
    "EMB-01": "Comparación de modelos de embeddings",
    "EMB-02": "Preprocesamiento de los embeddings (alineación, normalización, distancia)",
    "EMB-03": "Calidad de los embeddings según tamaño y pose del rostro",
    "CLU-01": "Agrupación de embeddings de rostros etiquetados (límite superior)",
    "CLU-02": "Agrupación de los rostros que entrega el pipeline",
    "CLU-03": "Agrupación con la regla de personas vistas en el mismo fotograma",
    "E2E-01": "Mejor configuración encadenada frente a la base",
    "E2E-03": "Configuración de producción (tracker A, 40 recortes, margen 0.8) con y sin MIN_FACE_PX 12 en los 3 videos",
    "E2E-02": "Configuración base frente a las ajustadas en los videos nuevos",
    "TRK-04": "Cuadrícula del tracker repetida en todos los videos con los recortes y el margen fijados",
    "CLU-04": "Personas dentro de un video (agrupación de producción: rostros agrupados y pistas unidas por mayoría)",
    "XV-01": "La misma persona en distintos videos (embeddings por persona, umbral de coincidencia para el backend)",
}
VERDICT_ES = {"PASS": "CUMPLE", "NEAR": "CERCA", "FAIL": "NO CUMPLE", "PENDING": "PENDIENTE", "no_target": "sin objetivo"}
VERDICT_FILL = {"CUMPLE": "C6EFCE", "CERCA": "FFEB9C", "NO CUMPLE": "FFC7CE"}
FINDING_COLUMNS = {"id": "id", "date": "fecha", "test_id": "prueba", "finding": "hallazgo",
                   "evidence": "evidencia", "consequence": "consecuencia"}


def read_csv(path):
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def item_sheet(df, rel):
    frames = []
    for _, r in df.iterrows():
        rows = read_csv(os.path.join(runlog.EXP_DIR, r["run_dir"], rel))
        if rows:
            t = pd.DataFrame(rows)
            t.insert(0, "run_label", r["label"])
            t.insert(0, "test_id", r["test_id"])
            t.insert(0, "run_id", r["run_id"])
            frames.append(t)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def summary(df):
    rows = []
    for test_id in sorted(set(df["test_id"]) | set(TEST_DESCRIPTIONS)):
        primary, higher, targets = registry.TESTS.get(test_id, (None, True, {}))
        sub = df[df["test_id"] == test_id] if not df.empty else df
        row = {"Prueba": test_id, "Descripción": TEST_DESCRIPTIONS.get(test_id, ""),
               "Ejecuciones": len(sub), "Ejecuciones ok": int((sub["status"] == "ok").sum()) if len(sub) else 0,
               "Ejecuciones que cumplen": int((sub["verdict"] == "PASS").sum()) if len(sub) else 0,
               "Métrica principal": primary or "", "Mejor si es": ("mayor" if higher else "menor") if primary else "",
               "Objetivos": "; ".join(f"{k} >= {v}" for k, v in targets.items()),
               "Ganador": "", "Valor del ganador": None, "Veredicto del ganador": "", "Valor de la base": None,
               "Cambio frente a la base": None, "Directorio del ganador": "", "Estado": "sin ejecutar" if not len(sub) else ""}
        col = f"metric.{primary}" if primary else None
        ok = sub[(sub["status"] == "ok")] if len(sub) else sub
        xv = cross_video(df, test_id) if len(sub) else pd.DataFrame()
        if col and not xv.empty and f"{col}.mean" in xv and xv[f"{col}.mean"].notna().any():
            full = xv[(xv["n_videos_scored"] == xv["n_videos_scored"].max()) & xv[f"{col}.mean"].notna()]
            best = full.loc[full[f"{col}.mean"].idxmax() if higher else full[f"{col}.mean"].idxmin()]
            unscored = best["n_videos"] - best["n_videos_scored"]
            row.update({"Ganador": f"{best['labels']} (promedio de {best['n_videos_scored']} videos calculados"
                                   + (f"; {unscored} sin calcular todavía)" if unscored else ")"),
                        "Valor del ganador": best[f"{col}.mean"], "Veredicto del ganador": "ver hoja X",
                        "Directorio del ganador": "", "Estado": "calculado (varios videos)"})
        elif col and col in ok and ok[col].notna().any():
            best = ok.loc[ok[col].idxmax() if higher else ok[col].idxmin()]
            row.update({"Ganador": best["label"], "Valor del ganador": best[col],
                        "Veredicto del ganador": VERDICT_ES.get(best["verdict"], best["verdict"]),
                        "Directorio del ganador": best["run_dir"], "Estado": "calculado"})
            base = ok[(ok["label"] == "baseline") & ok[col].notna()]
            if not base.empty:
                row["Valor de la base"] = base[col].iloc[0]
                row["Cambio frente a la base"] = best[col] - base[col].iloc[0]
        elif len(sub):
            missing = sorted({m for m in sub["missing"].dropna() if m})
            row["Estado"] = "en espera: " + " | ".join(missing) if missing else "sin métrica principal"
        rows.append(row)
    return pd.DataFrame(rows)


def config_key(row, params):
    return json.dumps({c: row[c] for c in params}, sort_keys=True, default=str)


def cross_video(df, test_id):
    sub = df[(df["test_id"] == test_id) & (df["status"] == "ok")]
    if sub["video"].nunique() < 2:
        return pd.DataFrame()
    primary, higher, _ = registry.TESTS.get(test_id, (None, True, {}))
    params = [c for c in sub.columns if c.startswith("param.") and not c.startswith("param.items.pipeline_run")
              and sub[c].astype(str).nunique() > 1]
    metrics = [c for c in sub.columns if c.startswith("metric.") and pd.api.types.is_numeric_dtype(sub[c])
               and sub[c].notna().any()]
    if "param.items.pipeline_run" in sub:
        def source_config(ref):
            c = runlog.read_json(os.path.join(runlog.EXP_DIR, str(ref), "config.json"), {}) if isinstance(ref, str) else {}
            return runlog.config_hash({k: v for k, v in c.items() if k != "video"}) if c else ""
        sub = pd.concat([sub, sub["param.items.pipeline_run"].map(source_config).rename("param.source_pipeline_config")], axis=1)
        params = params + ["param.source_pipeline_config"]
    rows = []
    for key, g in sub.groupby(sub.apply(lambda r: config_key(r, params), axis=1)):
        pcol = f"metric.{primary}" if primary else None
        row = {"config": key, "n_videos": g["video"].nunique(),
               "n_videos_scored": int(g.loc[g[pcol].notna(), "video"].nunique()) if pcol in g else 0,
               "videos": " | ".join(sorted(os.path.splitext(os.path.basename(v))[0] for v in g["video"])),
               "labels": " | ".join(g["label"].astype(str))}
        row.update({c: g[c].iloc[0] for c in params})
        for c in metrics:
            row[c + ".mean"] = g[c].mean()
            row[c + ".min" if higher else c + ".max"] = g[c].min() if higher else g[c].max()
        rows.append(row)
    out = pd.DataFrame(rows)
    col = f"metric.{primary}.mean" if primary else None
    if col in out:
        out = out.sort_values(["n_videos_scored", col], ascending=[False, not higher], na_position="last")
    return out


def per_test(df, test_id):
    sub = df[df["test_id"] == test_id].dropna(axis=1, how="all")
    params = [c for c in sub.columns if c.startswith("param.") and sub[c].astype(str).nunique() > 1]
    metrics = [c for c in sub.columns if c.startswith("metric.")]
    front = [c for c in ("run_id", "label", "status", "verdict", "rank_in_test", "best_so_far", "pareto")
             if c in sub.columns]
    tail = [c for c in sub.columns if c.startswith(("cache.", "device.")) or c in ("missing", "run_dir")]
    out = sub[front + params + metrics + tail]
    primary, higher, _ = registry.TESTS.get(test_id, (None, True, {}))
    if primary and f"metric.{primary}" in out:
        out = out.sort_values(f"metric.{primary}", ascending=not higher, na_position="last")
    return out


def ground_truth():
    frames_rows, face_rows, crop_rows, id_rows = [], [], [], []
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
                                  "auto_sources": json.dumps(f.get("auto_sources")) if f.get("auto_sources") else ""})
        per_video[stem] = counts
        for key, it in runlog.read_json(os.path.join(gt_dir, "crop_labels.json"), {"items": {}})["items"].items():
            crop_rows.append({"video": stem, "key": key, "frame_idx": it["frame_idx"], "box": json.dumps(it["box"]),
                              "label": it.get("label"), "suggested": it.get("suggested"),
                              "source": it.get("source"), "labeled_utc": it.get("labeled_utc")})
    for p in runlog.read_json(runlog.IDENTITIES, {"identities": []})["identities"]:
        row = {"identidad": p["id"], "cámara": p.get("camera", ""), "no_seguro_entre_videos": bool(p.get("uncertain")),
               "notas": p.get("notes", "")}
        for stem, counts in per_video.items():
            row[f"rostros_en_fotogramas_revisados.{stem}"] = counts.get(p["id"], 0)
        row["videos"] = sum(1 for counts in per_video.values() if counts.get(p["id"]))
        id_rows.append(row)
    return (pd.DataFrame(frames_rows), pd.DataFrame(face_rows), pd.DataFrame(id_rows), pd.DataFrame(crop_rows))


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
    rep = runlog.read_json(os.path.join(runlog.RUNS_DIR, "ENV-01", "report.json"))
    if rep:
        rows.append({"item": "ejecución ENV-01", "value": rep.get("started_utc")})
        rows.append({"item": "commit de git", "value": rep["git"]["commit"]})
        for k, v in rep["video"].items():
            rows.append({"item": f"video.{k}", "value": v})
        for venv, r in rep["venvs"].items():
            for k in ("python", "gpu", "driver"):
                if k in r:
                    rows.append({"item": f"{venv}.{k}", "value": r[k]})
            for k, v in (r.get("packages") or {}).items():
                rows.append({"item": f"{venv}.package.{k}", "value": v})
            for k, v in (r.get("probe") or {}).items():
                rows.append({"item": f"{venv}.{k}", "value": json.dumps(v) if isinstance(v, (list, dict)) else v})
    return pd.DataFrame(rows).rename(columns={"item": "elemento", "value": "valor"})


def readme(df):
    lines = [
        ("Pipeline de reconocimiento facial: resultados de las pruebas", ""),
        ("Generado (UTC)", runlog.utc_now()),
        ("Origen", "runs/<test_id>/<run_id>/ (run.json, config.json, events.jsonl, metrics.json, classification/)"),
        ("Se regenera con", "venvs/eval/bin/python -m harness.evaluate --all (o harness.export_excel)"),
        ("Valores", "Todos los valores los mide o calcula el sistema de pruebas; nada se escribe a mano. Cada fila "
                    "lleva run_id / run_dir para llegar a sus registros. El único texto escrito a mano es la redacción "
                    "de Hallazgos (findings.yaml) y de Decisiones (decisions.yaml); las cifras de Decisiones se toman "
                    "de las métricas de las ejecuciones."),
        ("Advertencia", "Las pruebas hechas con un solo video son provisionales (columnas single_video y videos_in_test). "
                        "rank_in_test y best_so_far se calculan por prueba y por video; las hojas X <prueba> promedian "
                        "cada configuración sobre los videos en que se ejecutó."),
        ("Privacidad", "Este libro no contiene imágenes. Las miniaturas quedan solo en results/reports/ (local)."),
        ("", ""),
        ("Hoja", "Contenido"),
        ("Decisiones", "Decisiones tomadas a partir de los resultados, su motivo (con las cifras medidas y la ejecución de la que salen) y cómo revisarlas"),
        ("Resumen", "Una fila por prueba: ejecuciones, ganador, valor, veredicto frente al objetivo y cambio frente a la base"),
        ("Todas las ejecuciones", "Cada ejecución: estado, veredicto, posición, todos los parámetros (param.*), métricas (metric.*), dispositivos y uso de caché"),
        ("T <prueba>", "Ejecuciones de una prueba: solo los parámetros que varían, ordenadas por la métrica principal"),
        ("X <prueba>", "Pruebas hechas en varios videos: cada configuración promediada sobre sus videos (promedio y peor caso)"),
        ("Detalle - *", "Resultado por elemento de cada ejecución (detecciones correctas, falsas y perdidas, candidatos, rostros elegidos, clusters, personas)"),
        ("GT - *", "Etiquetas manuales: fotogramas (avance de la revisión), rostros, personas y etiquetas de los rostros del pipeline"),
        ("Hallazgos", "Registro de hallazgos (findings.yaml)"),
        ("Entorno", "Máquina, GPU, versiones de paquetes y dispositivos de ENV-01"),
        ("", ""),
        ("Prefijo de columna", "Significado"),
        ("param.<etapa>.<nombre>", "Valor del parámetro usado en la ejecución (stage1 = tracking.py, tracker = yaml del tracker, stage2 = face_recognition.py)"),
        ("metric.<familia>.<nombre>", "Métrica (det = detección, trk = seguimiento, sel = selección, emb = embeddings, clu = agrupación, e2e = pipeline completo, ppl = personas dentro de un video, xv = entre videos, time = segundos)"),
        ("device.<parte>", "Dispositivo que usó realmente esa parte de la ejecución"),
        ("cache.<etapa>", "TRUE = la salida de la etapa (y su tiempo) se reutilizó de una ejecución idéntica anterior"),
        ("", ""),
        ("verdict", "CUMPLE = alcanza todos los objetivos; CERCA = hasta 0.02 por debajo; NO CUMPLE; PENDIENTE = la métrica todavía no se puede calcular; sin objetivo = la prueba no tiene objetivo"),
        ("status", "ok / failed (falló) / timeout (excedió el tiempo) / invalid (falta un registro o falló la evaluación)"),
        ("rank_in_test", "1 = mejor ejecución de su prueba según la métrica principal"),
        ("pareto", "TRUE = ninguna otra ejecución de la prueba es a la vez mejor y más rápida"),
        ("", ""),
        ("Prueba", "Métrica principal / objetivos"),
    ]
    for t, (primary, higher, targets) in sorted(registry.TESTS.items()):
        lines.append((t, f"{primary} (mejor si es {'mayor' if higher else 'menor'})"
                         + (f"; objetivos: {', '.join(f'{k} >= {v}' for k, v in targets.items())}" if targets else "")))
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
        if ws.title == "LEAME":
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
        ws.freeze_panes = "B2" if ws.title.startswith(("Todas las ejecuciones", "T ", "X ")) else "A2"
        if ws.max_row > 1:
            ws.auto_filter.ref = ws.dimensions
        for i, col in enumerate(ws.iter_cols(min_row=1, max_row=min(ws.max_row, 200)), start=1):
            width = max(len(str(c.value)) if c.value is not None else 0 for c in col)
            ws.column_dimensions[get_column_letter(i)].width = max(8, min(width + 2, 48))
        if ws.title == "Decisiones":
            for letter, width in (("A", 10), ("B", 30), ("C", 34), ("D", 110), ("E", 50)):
                ws.column_dimensions[letter].width = width
            for row in ws.iter_rows(min_row=2):
                for c in row:
                    c.alignment = Alignment(wrap_text=True, vertical="top")
        verdict_cols = [c.column for c in ws[1] if c.value in ("verdict", "Veredicto del ganador")]
        for vc in verdict_cols:
            for r in range(2, ws.max_row + 1):
                cell = ws.cell(r, vc)
                if cell.value in VERDICT_FILL:
                    cell.fill = PatternFill("solid", fgColor=VERDICT_FILL[cell.value])
                    cell.font = Font(name=FONT, size=10, bold=True)
    wb.save(path)


def export():
    reg = registry.rebuild()
    df = pd.read_csv(reg, keep_default_na=False, na_values=[""]) if os.path.getsize(reg) > 1 else pd.DataFrame()
    sheets = {"LEAME": readme(df), "Decisiones": decisions(df),
              "Resumen": summary(df) if not df.empty else pd.DataFrame(), "Todas las ejecuciones": df}
    for test_id in sorted(df["test_id"].unique()) if not df.empty else []:
        sheets[f"T {test_id}"] = per_test(df, test_id)
        xv = cross_video(df, test_id)
        if not xv.empty:
            sheets[f"X {test_id}"] = xv
    for name, rel in ITEM_FILES:
        sheets[name] = item_sheet(df, rel) if not df.empty else pd.DataFrame()
    gt_frames, gt_faces, gt_ids, gt_crops = ground_truth()
    sheets.update({"GT - fotogramas": gt_frames, "GT - rostros": gt_faces, "GT - personas": gt_ids,
                   "GT - rostros del pipeline": gt_crops})
    findings = yaml.safe_load(open(FINDINGS)) if os.path.exists(FINDINGS) else []
    sheets["Hallazgos"] = pd.DataFrame(findings or []).rename(columns=FINDING_COLUMNS)
    sheets["Entorno"] = environment()
    for name, frame in sheets.items():
        if "verdict" in frame.columns:
            sheets[name] = frame.assign(verdict=frame["verdict"].map(lambda v: VERDICT_ES.get(v, v)))
    os.makedirs(registry.RESULTS_DIR, exist_ok=True)
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
