import argparse
import csv
import glob
import os
from collections import Counter

from harness import runlog


def resolve(ref):
    if os.path.isdir(ref):
        return ref
    hits = glob.glob(os.path.join(runlog.RUNS_DIR, "*", ref))
    if len(hits) != 1:
        raise SystemExit(f"no se puede resolver la ejecución {ref!r}: {len(hits)} coincidencias")
    return hits[0]


def flat(d, prefix=""):
    out = {}
    for k, v in (d or {}).items():
        if isinstance(v, dict):
            out.update(flat(v, f"{prefix}{k}."))
        else:
            out[prefix + k] = v
    return out


def outcome_counts(run_dir):
    counts = {}
    for name, col in (("det_matches.csv", "outcome"), ("selection.csv", "class"),
                      ("cluster_errors.csv", "kind"), ("e2e_people.csv", "class")):
        path = os.path.join(run_dir, "classification", name)
        if os.path.exists(path) and os.path.getsize(path):
            with open(path) as f:
                for k, v in Counter(r[col] for r in csv.DictReader(f)).items():
                    counts[f"{name.split('.')[0]}:{k}"] = v
    return counts


def show(title, a, b, only_diff):
    keys = sorted(set(a) | set(b))
    rows = [(k, a.get(k), b.get(k)) for k in keys if not only_diff or a.get(k) != b.get(k)]
    if not rows:
        return
    print(f"\n## {title}")
    w = max(len(k) for k, _, _ in rows)
    for k, va, vb in rows:
        delta = ""
        if isinstance(va, (int, float)) and isinstance(vb, (int, float)) and not isinstance(va, bool):
            delta = f"  ({vb - va:+.4g})"
        print(f"{k:<{w}}  {va!s:>14}  {vb!s:>14}{delta}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    args = ap.parse_args()
    a, b = resolve(args.a), resolve(args.b)
    print(f"A = {runlog.rel_exp(a)}\nB = {runlog.rel_exp(b)}")
    show("Parámetros que difieren", flat(runlog.read_json(os.path.join(a, "config.json"))),
         flat(runlog.read_json(os.path.join(b, "config.json"))), True)
    ma = {k: v for k, v in runlog.read_json(os.path.join(a, "metrics.json"), {}).items() if not k.startswith("_")}
    mb = {k: v for k, v in runlog.read_json(os.path.join(b, "metrics.json"), {}).items() if not k.startswith("_")}
    show("Métricas", ma, mb, False)
    show("Resultados por elemento", outcome_counts(a), outcome_counts(b), False)


if __name__ == "__main__":
    main()
