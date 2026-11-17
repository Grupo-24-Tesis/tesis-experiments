import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))

import pytest

from bench import bench_compuerta


def entry(found=5, tp=90, fn=10, fp=5, wall=10.0):
    return {
        "duration_s": 5.0, "wall_s": wall, "factor": wall / 5.0, "analysed_frames": 50,
        "detections_file": "x/detections.json",
        "identity": {"labelled": 5, "found": found, "duplicates": 0, "mixed": 0, "purity": 1.0},
        "detection": {"tp": tp, "fn": fn, "fp": fp, "precision": tp / (tp + fp),
                      "recall": tp / (tp + fn), "f1": 0.9},
    }


def document(*entries):
    return {"videos": {f"v{i}": e for i, e in enumerate(entries)}}


def no_detections(document, entry):
    return []


def test_equal_runs_pass():
    base = document(entry(), entry())
    lines, code = bench_compuerta.compare_documents(base, base, 0.5, no_detections)
    assert code == 0
    assert any(line.startswith("Totales") for line in lines)


def test_recall_drop_over_threshold_fails():
    lines, code = bench_compuerta.compare_documents(
        document(entry(tp=90, fn=10)), document(entry(tp=85, fn=15)), 0.5, no_detections)
    assert code == 1
    assert any("REGRESIÓN" in line and "exhaustividad" in line for line in lines)


def test_recall_drop_under_threshold_passes():
    lines, code = bench_compuerta.compare_documents(
        document(entry(tp=1000, fn=10)), document(entry(tp=999, fn=11)), 0.5, no_detections)
    assert code == 0


def test_people_found_drop_fails_even_with_equal_recall():
    lines, code = bench_compuerta.compare_documents(
        document(entry(found=5), entry()), document(entry(found=4), entry()), 0.5, no_detections)
    assert code == 1
    assert any("personas encontradas" in line and "REGRESIÓN" in line for line in lines)


def test_missing_detections_are_listed():
    def loader(doc, e):
        return [{"frame": 1, "time_s": 3725.46, "box": [0, 0, 5, 5]}] if doc is base else []

    base = document(entry())
    candidate = document(entry())
    lines, code = bench_compuerta.compare_documents(base, candidate, 0.5, loader)
    assert any("01:02:05.4" in line and "[0, 0, 5, 5]" in line for line in lines)


def test_parse_env():
    assert bench_compuerta.parse_env(["A=1", "B=x=y"]) == {"A": "1", "B": "x=y"}
    with pytest.raises(SystemExit):
        bench_compuerta.parse_env(["A"])
