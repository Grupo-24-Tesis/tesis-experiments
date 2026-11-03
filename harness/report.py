import argparse
import base64
import csv
import json
import os

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from harness import registry, runlog
from harness.export_excel import VERDICT_ES

REPORTS = os.path.join(registry.RESULTS_DIR, "reports")
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
GALLERY_LIMIT = 60


def style(ax, title, xlabel, ylabel):
    ax.set_title(title, loc="left", fontsize=11, color=INK)
    ax.set_xlabel(xlabel, color=MUTED)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED)


def save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor="#fcfcfb")
    plt.close(fig)
    return os.path.basename(path)


def varying_params(df):
    cols = [c for c in df.columns if c.startswith("param.")]
    return [c for c in cols if df[c].astype(str).nunique() > 1]


def param_plots(df, primary, out_dir):
    files = []
    ok = df[(df["status"] == "ok") & df[f"metric.{primary}"].notna()]
    params = varying_params(ok)
    if len(params) == 2 and len(ok) == ok[params].astype(str).drop_duplicates().shape[0]:
        a, b = params
        pivot = ok.pivot_table(index=a, columns=b, values=f"metric.{primary}", aggfunc="mean")
        fig, ax = plt.subplots(figsize=(6, 4))
        im = ax.imshow(pivot.values, cmap="Blues", aspect="auto")
        ax.set_xticks(range(len(pivot.columns)), [str(c) for c in pivot.columns])
        ax.set_yticks(range(len(pivot.index)), [str(i) for i in pivot.index])
        for (i, j), v in np.ndenumerate(pivot.values):
            if not np.isnan(v):
                ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=8,
                        color="#ffffff" if v > np.nanmean(pivot.values) else INK)
        style(ax, f"{primary} en la cuadrícula", b.removeprefix("param."), a.removeprefix("param."))
        ax.grid(False)
        fig.colorbar(im, ax=ax, shrink=0.8)
        files.append(save(fig, os.path.join(out_dir, "grid.png")))
        return files
    for p in params:
        sub = ok.copy()
        sub["_x"] = pd.to_numeric(sub[p], errors="coerce")
        numeric = sub["_x"].notna().all()
        if not numeric:
            sub["_x"] = sub[p].astype(str)
        g = sub.groupby("_x")[f"metric.{primary}"].mean().sort_index()
        fig, ax = plt.subplots(figsize=(5, 3.2))
        xs = list(g.index) if numeric else list(range(len(g)))
        ax.plot(xs, g.values, color=SERIES[0], linewidth=2, marker="o", markersize=7)
        if not numeric:
            ax.set_xticks(xs, [str(x) for x in g.index])
        style(ax, f"{primary} según {p.removeprefix('param.')}", p.removeprefix("param."), primary)
        files.append(save(fig, os.path.join(out_dir, f"{p.removeprefix('param.').replace('.', '_')}.png")))
    return files


def read_csv(path):
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def curves(best_dir, out_dir):
    files = []
    pr = read_csv(os.path.join(best_dir, "classification", "pr_curve.csv"))
    if pr:
        t = [float(r["threshold"]) for r in pr]
        fig, ax = plt.subplots(figsize=(5, 3.2))
        ax.plot([float(r["recall"]) for r in pr], [float(r["precision"]) for r in pr],
                color=SERIES[0], linewidth=2)
        style(ax, "Precisión y recall (ejecución ganadora)", "recall", "precisión")
        files.append(save(fig, os.path.join(out_dir, "pr_curve.png")))
        fig, ax = plt.subplots(figsize=(5, 3.2))
        ax.plot(t, [float(r["f1"]) for r in pr], color=SERIES[0], linewidth=2)
        style(ax, "F1 según el umbral de confianza", "umbral", "F1")
        files.append(save(fig, os.path.join(out_dir, "f1_threshold.png")))
    dist = os.path.join(best_dir, "outputs", "distances.npz")
    if os.path.exists(dist):
        d = np.load(dist)
        fig, ax = plt.subplots(figsize=(5.5, 3.4))
        bins = np.linspace(0, max(d["genuine"].max(initial=0), d["impostor"].max(initial=0), 1e-6), 40)
        ax.hist(d["genuine"], bins=bins, color=SERIES[0], alpha=0.75, label="misma persona", density=True)
        ax.hist(d["impostor"], bins=bins, color=SERIES[1], alpha=0.6, label="personas distintas", density=True)
        metrics = runlog.read_json(os.path.join(best_dir, "metrics.json"), {})
        tau = metrics.get("emb.tau@far0.01")
        if tau is not None:
            ax.axvline(tau, color=INK, linewidth=1, linestyle="--")
            ax.text(tau, ax.get_ylim()[1] * 0.95, " τ @ FAR 1%", color=INK, fontsize=8, va="top")
        style(ax, "Distancias entre pares (ejecución ganadora)", "distancia", "densidad")
        ax.legend(frameon=False)
        files.append(save(fig, os.path.join(out_dir, "distances.png")))
    return files


def thumb(image, box=None, margin=0.6, size=96):
    if image is None:
        return None
    if box is not None:
        h, w = image.shape[:2]
        x1, y1, x2, y2 = box
        m = max(x2 - x1, y2 - y1) * margin
        image = image[int(max(0, y1 - m)):int(min(h, y2 + m)), int(max(0, x1 - m)):int(min(w, x2 + m))]
        if image.size == 0:
            return None
    k = size / max(image.shape[:2])
    image = cv2.resize(image, None, fx=k, fy=k, interpolation=cv2.INTER_AREA if k < 1 else cv2.INTER_CUBIC)
    ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode() if ok else None


def gallery(best_dir, out_dir, title):
    cfg = runlog.read_json(os.path.join(best_dir, "config.json"), {})
    stem = os.path.splitext(os.path.basename(cfg.get("video", "")))[0]
    frames_dir = os.path.join(runlog.GT_ROOT, stem, "frames")
    sections = []
    det = read_csv(os.path.join(best_dir, "classification", "det_matches.csv"))
    for outcome in ("FP", "FN"):
        cards = []
        for r in [r for r in det if r["outcome"] == outcome][:GALLERY_LIMIT]:
            box = json.loads(r["det_box"] if outcome == "FP" else r["gt_box"])
            src = thumb(cv2.imread(os.path.join(frames_dir, f"{int(r['frame_idx']):06d}.jpg")), box)
            if src:
                cards.append((src, f"fotograma {r['frame_idx']}, {r['size_bucket']} px"
                                   + (f", puntuación {float(r['score']):.2f}" if r["score"] else "")))
        if cards:
            sections.append((f"{outcome} ({'detecciones falsas' if outcome == 'FP' else 'rostros perdidos'})", cards))
    sel = read_csv(os.path.join(best_dir, "classification", "selection.csv"))
    if sel:
        faces = os.path.join(best_dir, "outputs", "faces", "candidate_faces")
        cards = [(thumb(cv2.imread(os.path.join(faces, r["track_id"], r["crop"]))),
                  f"pista {r['track_id']}, {r['class']}, {r['label'] or '?'}") for r in sel]
        sections.append(("Rostro elegido por pista", [c for c in cards if c[0]]))
    clusters = read_csv(os.path.join(best_dir, "outputs", "clusters.csv"))
    its = {it["item_id"]: it for it in runlog.read_json(os.path.join(best_dir, "outputs", "items.json"), [])}
    if clusters:
        by = {}
        for r in clusters:
            by.setdefault(r["cluster"], []).append(r)
        for k, rows in sorted(by.items(), key=lambda kv: int(kv[0])):
            cards = []
            for r in rows[:12]:
                it = its.get(r["item_id"], {})
                src = thumb(cv2.imread(runlog.exp_path(it["image"])) if it.get("image") else None, it.get("box"), margin=0.3)
                if src:
                    cards.append((src, r["identity"] or "?"))
            ids = sorted({r["identity"] or "?" for r in rows})
            sections.append((f"Grupo {k}, {len(rows)} rostros, {', '.join(ids)}"
                             + (", MEZCLA" if len([i for i in ids if i != '?']) > 1 else ""), cards))
    if not sections:
        return None
    html = ["<!doctype html><meta charset=utf-8><title>Galería de la ejecución</title><style>"
            "body{font:13px system-ui;margin:16px;background:#fcfcfb;color:#0b0b0b}"
            ".g{display:flex;flex-wrap:wrap;gap:8px}.c{width:110px;font-size:11px;color:#52514e}"
            "img{width:96px;height:96px;object-fit:contain;background:#000;border-radius:4px}</style>",
            f"<h1>{title}</h1><p>{runlog.rel_exp(best_dir)}</p>"]
    for head, cards in sections:
        html.append(f"<h2>{head}</h2><div class=g>")
        html += [f"<div class=c><img src='{src}'><div>{cap}</div></div>" for src, cap in cards]
        html.append("</div>")
    path = os.path.join(out_dir, "gallery.html")
    with open(path, "w") as f:
        f.write("\n".join(html))
    return "gallery.html"


def fmt(v):
    if isinstance(v, float):
        return f"{v:.4f}"
    return "" if v is None or (isinstance(v, float) and np.isnan(v)) else str(v)


def report(df, test_id):
    primary, higher, targets = registry.TESTS.get(test_id, (None, True, {}))
    sub = df[df["test_id"] == test_id]
    out_dir = os.path.join(REPORTS, test_id)
    os.makedirs(out_dir, exist_ok=True)
    lines = [f"# {test_id}", "", f"- Ejecuciones: {len(sub)} ({(sub['status'] == 'ok').sum()} ok)",
             f"- Métrica principal: `{primary}` (mejor si es {'mayor' if higher else 'menor'})",
             f"- Objetivos: {targets or 'ninguno'}",
             f"- Videos: {sub['videos_in_test'].max() if 'videos_in_test' in sub else 1}"
             + (" (un solo video: conclusiones provisionales)" if sub.get("single_video", True).all() else ""), ""]
    col = f"metric.{primary}" if primary else None
    ok = sub[sub["status"] == "ok"]
    best_dir = None
    if col in sub and ok[col].notna().any():
        ranked = ok[ok[col].notna()].sort_values(col, ascending=not higher)
        best = ranked.iloc[0]
        best_dir = os.path.join(runlog.EXP_DIR, best["run_dir"])
        base = ok[ok["label"] == "baseline"]
        lines += [f"**Ganador:** `{best['label']}`, {primary} = {fmt(best[col])}, veredicto **{VERDICT_ES.get(best['verdict'], best['verdict'])}**,"
                  f" `{best['run_dir']}`"]
        if not base.empty and base[col].notna().any():
            b = base[col].dropna().iloc[0]
            lines.append(f"**Base:** {fmt(b)}, cambio {best[col] - b:+.4f}")
        params = varying_params(ok)
        show = ["label", col, "verdict", "metric.time.total_s"] + params
        show = [c for c in show if c in ranked]
        lines += ["", "## Mejores ejecuciones", "", "| " + " | ".join(c.removeprefix("metric.").removeprefix("param.")
                                                          for c in show) + " |",
                  "|" + "---|" * len(show)]
        for _, r in ranked.head(10).iterrows():
            lines.append("| " + " | ".join(fmt(r[c]) for c in show) + " |")
        plots = param_plots(sub, primary, out_dir) + curves(best_dir, out_dir)
        if plots:
            lines += ["", "## Gráficos", ""] + [f"![{p}]({p})" for p in plots]
    else:
        lines.append("Todavía no hay ejecuciones calculadas (faltan etiquetas o fallaron las ejecuciones). Falta:")
        lines += [f"- {m}" for m in sub["missing"].dropna().unique() if m]
    failed = sub[sub["status"] != "ok"]
    if not failed.empty:
        lines += ["", "## Ejecuciones fallidas o inválidas", ""] + [
            f"- `{r['run_dir']}`: {r['status']} {r.get('problems', '')}" for _, r in failed.iterrows()]
    if best_dir and gallery(best_dir, out_dir, f"{test_id}: ganador"):
        lines += ["", "Galería de la ejecución ganadora: [gallery.html](gallery.html) (solo local, contiene rostros)"]
    path = os.path.join(out_dir, "report.md")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return path, (test_id, sub, primary, col)


def summary(df, results):
    lines = ["# Resumen de resultados", "", f"Generado el {runlog.utc_now()}", "",
             "| Prueba | Ejecuciones | Ganador | Métrica principal | Valor | Veredicto | Informe |", "|---|---|---|---|---|---|---|"]
    for test_id, sub, primary, col in results:
        ok = sub[sub["status"] == "ok"]
        if col and col in ok and ok[col].notna().any():
            higher = registry.TESTS.get(test_id, (None, True, {}))[1]
            best = ok.loc[ok[col].idxmax() if higher else ok[col].idxmin()]
            lines.append(f"| {test_id} | {len(sub)} | {best['label']} | {primary} | {fmt(best[col])} | "
                         f"{VERDICT_ES.get(best['verdict'], best['verdict'])} | [informe](reports/{test_id}/report.md) |")
        else:
            lines.append(f"| {test_id} | {len(sub)} | - | {primary} | - | PENDIENTE | "
                         f"[informe](reports/{test_id}/report.md) |")
    path = os.path.join(registry.RESULTS_DIR, "summary.md")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-id")
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()
    df = pd.read_csv(registry.rebuild())
    tests = sorted(df["test_id"].unique()) if args.all else [args.test_id]
    results = []
    for t in tests:
        path, res = report(df, t)
        results.append(res)
        print(runlog.rel_exp(path))
    if args.all:
        print(runlog.rel_exp(summary(df, results)))


if __name__ == "__main__":
    main()
