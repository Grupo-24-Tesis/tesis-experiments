import json
import os

import pytest

from harness import flujo_actual


def write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(text)


def detection(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def identity(labelled, found, duplicates, mixed, purity, matched):
    return {"labelled": labelled, "identities": found + duplicates, "found": found,
            "duplicates": duplicates, "mixed": mixed, "without_person": 0, "purity": purity,
            "faces_total": matched, "faces_matched": matched}


def entry(video, wall, duration, frames, gated, det, ident, kind="chokepoint"):
    result = {"video": video, "duration_s": duration, "wall_s": wall, "factor": wall / duration,
              "analysed_frames": frames, "detections_count": det["tp"] + det["fp"],
              "identities_count": ident["identities"], "kind": kind, "identity": ident,
              "detection": det, "detections_file": f"{os.path.basename(video)[:-4]}/detections.json"}
    if gated:
        result.update({"gated": True, "gate_wall_s": 1.0, "face_wall_s": wall - 1.0,
                       "active_fraction": 0.5, "switch_count": 2})
    return result


def detections(frames, boxes):
    return json.dumps({"fps": 25.0, "frames_analysed": frames,
                       "detections": [{"frame": f, "time_s": f / 25.0, "box": b} for f, b in boxes]})


@pytest.fixture
def run_dir(tmp_path, monkeypatch):
    root = tmp_path / "flujo"
    box = [10, 10, 50, 50]
    videos = {
        "P1E_S1_C1": "/data/x/chokepoint/videos/P1E_S1_C1.mp4",
        "Camera3": "/data/x/experiments/videos/Camera3.mp4",
        "largo": "/data/x/chokepoint/runs/gating/videos_prueba/largo.mp4",
    }
    gated, base = {"videos": {}}, {"videos": {}}
    for name, video in videos.items():
        kind = "cctv" if name == "Camera3" else "chokepoint"
        gated["videos"][name] = entry(video, 10.0, 20.0, 5, True, detection(8, 2, 2),
                                      identity(4, 3, 1, 0, 0.9, 10), kind)
        base["videos"][name] = entry(video, 40.0, 20.0, 10, False, detection(9, 2, 1),
                                     identity(4, 4, 0, 0, 0.95, 10), kind)
        write(str(root / "compuerta_defecto" / name / "detections.json"),
              detections([0, 1], [(0, box)]))
        write(str(root / "sin_compuerta" / name / "detections.json"),
              detections([0, 1, 2], [(0, box), (2, box)]))
    write(str(root / "compuerta_defecto" / "results.json"), json.dumps(gated))
    write(str(root / "sin_compuerta" / "results.json"), json.dumps(base))
    write(str(root / "estado.txt"),
          "pruebas 0 2026-01-01T00:00:00Z 2026-01-01T00:00:05Z\n"
          "entorno 0 2026-01-01T00:00:05Z 2026-01-01T00:01:00Z\n"
          "compuerta 0 2026-01-01T00:01:00Z 2026-01-01T01:00:00Z\n"
          "base 0 2026-01-01T01:00:00Z 2026-01-01T02:00:00Z\n"
          "comparacion 1 2026-01-01T02:00:00Z 2026-01-01T02:01:00Z\n")
    write(str(root / "versiones.txt"), "pipeline abc test-suite\npipeline_cambios 0\n")
    write(str(root / "pruebas.log"), "....\n451 passed, 2 warnings in 3.72s\n")
    write(str(root / "comparacion.log"), "Totales\nREGRESIÓN: x: personas encontradas 4 -> 3\n")
    monkeypatch.setattr(flujo_actual.bench_compuerta, "gated_detection_over_baseline",
                        lambda name, base_doc, cand_doc: (detection(7, 2, 3)
                                                          if name != "Camera3" else None))
    return str(root)


def test_dataset_of_classifies_by_path():
    assert flujo_actual.dataset_of({"video": "/d/chokepoint/videos/a.mp4"}) == "ChokePoint"
    assert flujo_actual.dataset_of(
        {"video": "/d/chokepoint/runs/gating/videos_prueba/a.mp4"}) == "clips de la compuerta"
    assert flujo_actual.dataset_of({"video": "/d/experiments/videos/a.mp4"}) == "residencial"


def test_chain_state_reads_steps_and_versions(run_dir):
    steps, versions = flujo_actual.chain_state(run_dir)
    assert steps["comparacion"] == {"codigo": 1, "inicio": "2026-01-01T02:00:00Z",
                                    "fin": "2026-01-01T02:01:00Z"}
    assert versions == {"pipeline": "abc test-suite", "pipeline_cambios": "0"}


def test_pytest_counts_and_regressions(run_dir):
    assert flujo_actual.pytest_counts(run_dir) == {"passed": 451}
    assert flujo_actual.regressions(run_dir) == ["x: personas encontradas 4 -> 3"]


def test_video_rows_compare_gated_with_base(run_dir):
    gated, base, comparisons = flujo_actual.load_runs(run_dir)
    rows = flujo_actual.video_rows(gated, base, comparisons).set_index("video")
    assert rows.loc["P1E_S1_C1", "comparacion.detecciones_base_ausentes"] == 1
    assert rows.loc["P1E_S1_C1", "comparacion.exhaustividad_compuerta_sobre_base"] == 0.7
    assert rows.loc["P1E_S1_C1", "comparacion.tiempo_relativo"] == 0.25
    assert rows.loc["Camera3", "conjunto"] == "residencial"
    assert rows.loc["Camera3", "compuerta.encontradas"] == 3


def test_total_rows_group_by_dataset(run_dir):
    gated, base, comparisons = flujo_actual.load_runs(run_dir)
    totals = flujo_actual.total_rows(gated, base, comparisons).set_index("conjunto")
    assert list(totals.index) == ["residencial", "ChokePoint", "clips de la compuerta", "todos"]
    everything = totals.loc["todos"]
    assert everything["videos"] == 3
    assert everything["comparacion.detecciones_base_ausentes"] == 3
    assert everything["compuerta.factor"] == pytest.approx(0.5)
    assert everything["base.factor"] == pytest.approx(2.0)
    assert everything["compuerta.exhaustividad"] == pytest.approx(0.8)
    assert everything["comparacion.exhaustividad_compuerta_sobre_base"] == pytest.approx(22 / 30)
    assert everything["comparacion.caida_exhaustividad_pp"] == pytest.approx((0.9 - 22 / 30) * 100)
    assert everything["compuerta.etiquetadas"] == 12


def cross_document():
    block = {"metricas": {"consultas": 10, "acierto_1": 0.9, "acierto_5": 1.0,
                          "azar": {"acierto_1": 0.03}},
             "umbrales": [{"umbral": 0.5, "precision": 0.8, "exhaustividad": 0.95},
                          {"umbral": 0.55, "precision": 0.9, "exhaustividad": 0.9}],
             "sin_objetivo": 0}
    return {"extraccion": {"videos": 72, "identidades": 100, "videos_distintos": ["a"]},
            "resultados": {"media": {"todas": block, "otro_portal": block},
                           "mejor": {"todas": block}}}


def test_cross_camera_rows_flatten_chance():
    rows = flujo_actual.cross_camera_rows(cross_document())
    first = rows.iloc[0]
    assert first["representacion"] == "media de la identidad"
    assert first["caso"] == "todas las demás cámaras"
    assert first["acierto_1"] == 0.9
    assert first["azar.acierto_1"] == 0.03
    assert len(flujo_actual.threshold_rows(cross_document())) == 4
    assert flujo_actual.extraction_info(cross_document())["videos_distintos"] == 1


def test_summary_rows_report_each_test(run_dir):
    gated, base, comparisons = flujo_actual.load_runs(run_dir)
    videos = flujo_actual.video_rows(gated, base, comparisons)
    totals = flujo_actual.total_rows(gated, base, comparisons)
    document = cross_document()
    summary = flujo_actual.summary_rows(
        run_dir, {"metric.env.pass_rate": 1.0, "metric.env.passed": 15, "metric.env.checks": 15},
        totals, videos, flujo_actual.cross_camera_rows(document),
        flujo_actual.threshold_rows(document), flujo_actual.extraction_info(document))
    rows = summary.set_index("Prueba")
    assert list(rows.index) == ["PRU-01", "ENV-01", "FLU-01", "FLU-02", "FLU-03"]
    assert rows.loc["PRU-01", "Valor"] == 451
    assert rows.loc["ENV-01", "Valor"] == 1.0
    assert rows.loc["FLU-01", "Valor"] == pytest.approx(0.8)
    assert rows.loc["FLU-02", "Valor"] == 3
    assert rows.loc["FLU-02", "Código de salida"] == 1
    assert rows.loc["FLU-02", "Inicio (UTC)"] == "2026-01-01T01:00:00Z"
    assert "regresiones 1" in rows.loc["FLU-02", "Otras cifras"]
    assert rows.loc["FLU-03", "Valor"] == 0.9
    assert "precisión 90,0 %" in rows.loc["FLU-03", "Otras cifras"]
    assert rows.loc["FLU-03", "Código de salida"] is None
