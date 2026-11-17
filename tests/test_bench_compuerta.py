import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))

import pytest

from bench import bench_compuerta


def entry(found=5, tp=90, fn=10, fp=5, wall=10.0):
    return {
        "duration_s": 5.0, "wall_s": wall, "factor": wall / 5.0, "analysed_frames": 50,
        "detections_file": "x/detections.json",
        "identity": {"labelled": 5, "faces_matched": 10, "found": found, "duplicates": 0, "mixed": 0, "purity": 1.0},
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


def test_extra_arg_accepts_flag_with_equals():
    args = bench_compuerta.build_parser().parse_args(
        ["run", "--label", "a", "--out", "o", "--extra-arg=--no-gating", "--extra-arg=--x=1", "v.mp4"])
    assert args.extra_arg == ["--no-gating", "--x=1"]
    assert args.videos == ["v.mp4"]


def test_video_missing_in_candidate_fails():
    lines, code = bench_compuerta.compare_documents(
        document(entry(), entry()), document(entry()), 0.5, no_detections)
    assert code == 1
    assert any("ausente en el candidato" in line for line in lines)


def test_totals_use_only_videos_ok_in_both():
    failed = {"error": "x", "wall_s": 1.0}
    base = document(entry(wall=10.0), entry(wall=20.0))
    cand = document(entry(wall=12.0), failed)
    lines, code = bench_compuerta.compare_documents(base, cand, 0.5, no_detections)
    assert code == 1
    assert any("Excluido de los totales: v1" in line for line in lines)
    assert any("videos comparados: 1" in line for line in lines)
    assert any("tiempo de pared: 10.0 s -> 12.0 s" in line for line in lines)


def test_totals_include_detection_and_identity_summaries():
    base = document(entry(found=5, tp=90, fn=10, fp=10), entry(found=4, tp=10, fn=0, fp=0))
    lines, code = bench_compuerta.compare_documents(base, base, 0.5, no_detections)
    text = "\n".join(lines)
    assert "precisión total: 90.9% -> 90.9%" in text
    assert "exhaustividad total: 90.9% -> 90.9%" in text
    assert "personas encontradas total: 9 -> 9" in text
    assert "duplicados total: 0 -> 0" in text
    assert "mezcladas total: 0 -> 0" in text
    assert "pureza total: 100.0% -> 100.0%" in text


PRESENCE = {
    "fps": 10.0,
    "intervals": [[100, 200], [500, 600]],
    "switches": [{"frame": 100}, {"frame": 201}, {"frame": 500}, {"frame": 601}],
    "active_fraction": 0.2,
}


def test_gated_flag_is_parsed():
    parser = bench_compuerta.build_parser()
    assert parser.parse_args(["run", "--label", "a", "--out", "o", "--gated", "v.mp4"]).gated
    assert not parser.parse_args(["run", "--label", "a", "--out", "o", "v.mp4"]).gated


def test_gate_fields_record_times_fraction_and_switches():
    fields = bench_compuerta.gate_fields(PRESENCE, 12.5, 30.0)
    assert fields == {"gated": True, "gate_wall_s": 12.5, "face_wall_s": 30.0,
                      "active_fraction": 0.2, "switch_count": 4,
                      "presence_file": "presence.json"}


@pytest.mark.parametrize("frame,omitted", [(99, True), (100, False), (200, False), (201, True),
                                           (350, True), (500, False), (601, True)])
def test_omission_info_classifies_the_frame_against_the_intervals(frame, omitted):
    assert bench_compuerta.omission_info({"frame": frame}, PRESENCE)["omitted"] is omitted


def test_omission_info_gives_the_distance_in_seconds_to_the_nearest_switch():
    assert bench_compuerta.omission_info({"frame": 130}, PRESENCE)["distance_s"] == 3.0
    assert bench_compuerta.omission_info({"frame": 190}, PRESENCE)["distance_s"] == 1.1
    assert bench_compuerta.omission_info({"frame": 350}, PRESENCE)["distance_s"] == 14.9


def test_omission_info_without_switches_has_no_distance():
    presence = {"fps": 10.0, "intervals": [], "switches": []}
    info = bench_compuerta.omission_info({"frame": 5}, presence)
    assert info == {"omitted": True, "distance_s": None}


def gated_entry(**kwargs):
    e = entry(**kwargs)
    e.update({"gated": True, "gate_wall_s": 2.0, "face_wall_s": 4.0, "wall_s": 6.0,
              "active_fraction": 0.2, "switch_count": 4, "presence_file": "presence.json"})
    return e


def test_compare_shows_gate_time_total_factor_and_active_fraction():
    lines, code = bench_compuerta.compare_documents(
        document(entry(wall=12.0)), document(gated_entry()), 0.5, no_detections)
    text = "\n".join(lines)
    assert "compuerta: 2.0 s, rostros: 4.0 s, total 6.0 s" in text
    assert "factor total 0.50 del tiempo de la base" in text
    assert "tiempo activo: 20.0% del video, 4 cambios" in text
    assert "fracción analizada 1.00" in text


def test_compare_classifies_each_missing_detection_against_the_candidate_intervals():
    detections = {"base": [{"frame": 130, "time_s": 13.0, "box": [0, 0, 5, 5]},
                           {"frame": 350, "time_s": 35.0, "box": [1, 1, 6, 6]}]}

    def loader(doc, e):
        return detections["base"] if doc is base else []

    base = document(entry())
    candidate = document(gated_entry())
    lines, _ = bench_compuerta.compare_documents(base, candidate, 0.5, loader,
                                                 lambda doc, e: PRESENCE)
    text = "\n".join(lines)
    assert "dentro de un tramo activo, a 3.0 s del cambio más cercano" in text
    assert "tramo omitido, a 14.9 s del cambio más cercano" in text


def test_compare_of_ungated_runs_adds_no_gate_lines():
    lines, _ = bench_compuerta.compare_documents(document(entry()), document(entry()), 0.5,
                                                 no_detections)
    assert not any("compuerta" in line or "tramo" in line for line in lines)


def test_analysed_fraction_label_only_for_gated_candidates():
    gated, _ = bench_compuerta.compare_documents(
        document(entry()), document(gated_entry()), 0.5, no_detections)
    plain, _ = bench_compuerta.compare_documents(
        document(entry()), document(entry()), 0.5, no_detections)
    assert any("fracción analizada" in line for line in gated)
    assert not any("fracción analizada" in line for line in plain)
    assert any("(razón " in line for line in plain)
