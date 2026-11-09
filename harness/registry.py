import argparse
import glob
import os

import pandas as pd

from harness import runlog

RESULTS_DIR = runlog.RESULTS_DIR
REGISTRY = os.path.join(RESULTS_DIR, "registry.csv")
REQUIRED = ("run.json", "config.json", "events.jsonl")
NEAR_MARGIN = 0.02
DEVICE_ES = {"CPU (CUDA EP failed to load)": "CPU (no cargó el proveedor CUDA de ONNX)",
             "CPU (TF GPU libraries missing)": "CPU (faltan las librerías GPU de TensorFlow)"}
PIPELINE_REWRITTEN = {"ff1511563d": "c87c7cbc22", "1e38058b31": "c87c7cbc22", "3bed659b39": "514a88baf4"}
EXPERIMENTS_REWRITTEN = {"c1982c71d9": "07aae05c1e"}

TESTS = {
    "ENV-01": ("env.pass_rate", True, {"env.pass_rate": 1.0}),
    "DET-01": ("det.f1", True, {"det.f1": 0.85}),
    "DET-02": ("det.best_f1", True, {}),
    "DET-03": ("det.f1", True, {}),
    "DET-04": ("det.f1", True, {"det.f1": 0.85}),
    "DET-05": ("det.f1", True, {"det.f1": 0.85}),
    "EMB-00": ("emb0.control_gap", True, {}),
    "EMB-01": ("emb.tar@far0.01", True, {"emb.tar@far0.01": 0.90}),
    "EMB-02": ("emb.tar@far0.01", True, {}),
    "EMB-03": ("emb.tar@far0.01", True, {}),
    "CLU-01": ("clu.ari", True, {"clu.purity": 0.85, "clu.ari": 0.70}),
    "CLU-02": ("clu.ari", True, {"clu.purity": 0.85, "clu.ari": 0.70}),
    "CLU-03": ("clu.ari", True, {"clu.purity": 0.85, "clu.ari": 0.70}),
    "TRK-01": ("trk.coverage", True, {}),
    "TRK-02": ("trk.fragmentation", False, {}),
    "TRK-03": ("trk.best_face_quality_median", True, {}),
    "SEL-01": ("sel.valid_rate", True, {}),
    "E2E-01": ("e2e.tracks_identity_recall", True, {"e2e.tracks_identity_recall": 0.90}),
    "E2E-03": ("e2e.tracks_identity_recall", True, {"e2e.tracks_identity_recall": 0.90}),
    "E2E-02": ("e2e.tracks_identity_recall", True, {"e2e.tracks_identity_recall": 0.90}),
    "TRK-04": ("trk.fragmentation", False, {}),
    "CLU-04": ("ppl.person_f1", True, {"ppl.identity_recall": 0.90, "ppl.person_precision": 0.90,
                                       "ppl.merge_free": 0.95}),
    "XV-01": ("xv.open_set_accuracy", True, {"xv.open_set_accuracy": 0.90}),
    "ENV-02": ("sel.n_faces", True, {}),
    "ENV-03": ("sel.n_faces", True, {}),
}


def flatten(prefix, d, out):
    for k, v in d.items():
        if isinstance(v, dict):
            flatten(f"{prefix}{k}.", v, out)
        else:
            out[prefix + k] = v
    return out


def commit_id(commit, rewritten=None):
    commit = (commit or "")[:10]
    if commit in ("", "HEAD"):
        return "sin commits"
    return (rewritten or {}).get(commit, commit)


def git_columns(git):
    if "pipeline" not in git:
        return {"git_pipeline_commit": commit_id(git.get("commit"), PIPELINE_REWRITTEN),
                "git_pipeline_dirty": git.get("dirty"),
                "git_experiments_commit": "sin commits", "git_experiments_dirty": None}
    return {"git_pipeline_commit": commit_id(git["pipeline"].get("commit"), PIPELINE_REWRITTEN),
            "git_pipeline_dirty": git["pipeline"].get("dirty"),
            "git_experiments_commit": commit_id(git["experiments"].get("commit"), EXPERIMENTS_REWRITTEN),
            "git_experiments_dirty": git["experiments"].get("dirty")}


def validate(run_dir):
    problems = [f for f in REQUIRED if not os.path.exists(os.path.join(run_dir, f))]
    meta = runlog.read_json(os.path.join(run_dir, "run.json"), {})
    if meta.get("status") in (None, "running"):
        problems.append("estado sin definir (falló o sigue en curso)")
    if meta.get("status") == "ok" and not os.path.exists(os.path.join(run_dir, "metrics.json")):
        problems.append("falta metrics.json (ejecutar evaluate)")
    err = runlog.read_json(os.path.join(run_dir, "metrics.json"), {}).get("_error")
    if err:
        problems.append(f"falló la evaluación: {err}")
    return problems


def verdict(row, targets):
    if not targets:
        return "no_target"
    results = []
    for metric, target in targets.items():
        v = row.get("metric." + metric)
        if v is None or pd.isna(v):
            return "FAIL" if row.get("unmet") else "PENDING"
        results.append("PASS" if v >= target else ("NEAR" if v >= target - NEAR_MARGIN else "FAIL"))
    if all(r == "PASS" for r in results):
        return "PASS"
    return "FAIL" if "FAIL" in results else "NEAR"


def rebuild():
    rows = []
    for run_dir in sorted(glob.glob(os.path.join(runlog.RUNS_DIR, "*", "*"))):
        if "/_" in run_dir or not os.path.isdir(run_dir):
            continue
        meta = runlog.read_json(os.path.join(run_dir, "run.json"))
        if not meta or "run_id" not in meta:
            continue
        cfg = runlog.read_json(os.path.join(run_dir, "config.json"), {})
        metrics = runlog.read_json(os.path.join(run_dir, "metrics.json"), {})
        problems = validate(run_dir)
        status = meta.get("status")
        if status == "ok" and problems:
            status = "invalid"
        row = {"run_id": meta["run_id"], "test_id": meta["test_id"], "label": meta.get("label"),
               "repeat": meta.get("repeat"), "status": status, "problems": "; ".join(problems),
               "started_utc": meta.get("started_utc"), **git_columns(meta.get("git", {})),
               "video": " + ".join(meta["videos"]) if meta.get("videos") else meta.get("video", {}).get("path"),
               "run_dir": runlog.rel_exp(run_dir)}
        flatten("param.", {k: v for k, v in cfg.items() if k != "video"}, row)
        for k, v in metrics.items():
            if not k.startswith("_"):
                row["metric." + k] = v
        row["missing"] = "; ".join(metrics.get("_missing", []))
        row["unmet"] = metrics.get("_unmet", "")
        for k, v in meta.get("devices_used", {}).items():
            row["device." + k] = DEVICE_ES.get(v, v)
        for k, v in meta.get("cache_hits", {}).items():
            row["cache." + k] = v
        rows.append(row)
    df = pd.DataFrame(rows)
    if not df.empty:
        df["verdict"] = "no_target"
        df["rank_in_test"] = None
        df["pareto"] = False
        df["best_so_far"] = False
        df["videos_in_test"] = df.groupby("test_id")["video"].transform(
            lambda v: len({x for s in v.dropna() for x in s.split(" + ")}))
        df["single_video"] = df["videos_in_test"] <= 1
        for test_id, idx in df.groupby("test_id").groups.items():
            df.loc[idx, "verdict"] = [verdict(r, TESTS.get(test_id, (None, True, {}))[2])
                                      for _, r in df.loc[idx].iterrows()]
        for (test_id, _), idx in df.groupby(["test_id", "video"]).groups.items():
            primary, higher, targets = TESTS.get(test_id, (None, True, {}))
            sub = df.loc[idx]
            col = "metric." + primary if primary else None
            ok = sub[(sub["status"] == "ok") & sub[col].notna()] if col in sub else sub.iloc[0:0]
            if ok.empty:
                continue
            df.loc[ok.index, "rank_in_test"] = ok[col].rank(ascending=not higher, method="min").astype(int)
            df.loc[ok[col].idxmax() if higher else ok[col].idxmin(), "best_so_far"] = True
            t = ok.get("metric.time.total_s")
            if t is not None:
                for i, r in ok.iterrows():
                    better = ok[(ok[col] > r[col]) if higher else (ok[col] < r[col])]
                    df.loc[i, "pareto"] = not ((better["metric.time.total_s"] <= r["metric.time.total_s"]).any())
    os.makedirs(RESULTS_DIR, exist_ok=True)
    front = [c for c in ("run_id", "test_id", "label", "repeat", "status", "verdict", "rank_in_test",
                         "best_so_far", "pareto", "single_video", "videos_in_test") if c in df.columns]
    df = df[front + [c for c in df.columns if c not in front]] if not df.empty else df
    df.to_csv(REGISTRY, index=False)
    return REGISTRY


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true")
    args = ap.parse_args()
    if args.validate:
        bad = 0
        for run_dir in sorted(glob.glob(os.path.join(runlog.RUNS_DIR, "*", "*"))):
            if "/_" in run_dir or not os.path.exists(os.path.join(run_dir, "run.json")):
                continue
            problems = validate(run_dir)
            if problems:
                bad += 1
                print(f"INVÁLIDA {runlog.rel_exp(run_dir)}: {'; '.join(problems)}")
        print(f"{bad} ejecuciones inválidas")
        raise SystemExit(1 if bad else 0)
    print(rebuild())


if __name__ == "__main__":
    main()
