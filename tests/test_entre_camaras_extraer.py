import json
import os

import numpy as np
import pytest

from bench import entre_camaras, entre_camaras_extraer as extraer


def touch(path, text=""):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(text)


def complete_source(gating_root, run, name):
    touch(os.path.join(gating_root, run, name, "presence.json"), "{}")
    touch(os.path.join(gating_root, run, name, "identities.json"), "{}")


def test_chokepoint_videos_needs_video_and_groundtruth(tmp_path):
    touch(str(tmp_path / "videos" / "A.mp4"))
    touch(str(tmp_path / "groundtruth" / "A.xml"))
    touch(str(tmp_path / "videos" / "B.mp4"))
    touch(str(tmp_path / "groundtruth" / "C.xml"))
    assert extraer.chokepoint_videos(str(tmp_path)) == ["A"]


def test_source_run_prefers_defecto(tmp_path):
    root = str(tmp_path)
    complete_source(root, "compuerta_defecto", "A")
    complete_source(root, "compuerta_nuevos", "A")
    complete_source(root, "compuerta_nuevos", "B")
    touch(os.path.join(root, "compuerta_defecto", "C", "presence.json"), "{}")
    assert extraer.source_run("A", root) == os.path.join(root, "compuerta_defecto", "A")
    assert extraer.source_run("B", root) == os.path.join(root, "compuerta_nuevos", "B")
    assert extraer.source_run("C", root) is None


def test_plan_videos_reports_all_missing_sources(tmp_path):
    root = str(tmp_path / "gating")
    complete_source(root, "compuerta_defecto", "A")
    with pytest.raises(SystemExit) as error:
        extraer.plan_videos(["A", "B", "C"], root, str(tmp_path / "out"))
    assert "B" in str(error.value) and "C" in str(error.value)


def test_plan_videos_skips_finished(tmp_path):
    root = str(tmp_path / "gating")
    out = str(tmp_path / "out")
    for name in ("A", "B"):
        complete_source(root, "compuerta_defecto", name)
    touch(os.path.join(out, "A", "extraccion.json"), "{}")
    touch(os.path.join(out, "B", "embeddings.npz"))
    plan = extraer.plan_videos(["A", "B"], root, out)
    assert plan == [("B", os.path.join(root, "compuerta_defecto", "B"))]


def test_adaptive_sampling_is_refused():
    with pytest.raises(SystemExit) as error:
        extraer.check_settings(True, refusal=lambda gating, adaptive: "texto")
    assert "texto" in str(error.value)
    assert extraer.check_settings(False, refusal=lambda g, a: None) is None


def record(identity_id, boxes):
    return {"id": identity_id,
            "faces": [{"frame": 3 * k, "box": box} for k, box in enumerate(boxes)]}


def test_compare_identities():
    stored = {"identities": [record(1, [[0, 0, 5, 5], [1, 1, 6, 6]]),
                             record(2, [[2, 2, 7, 7]])]}
    same = [record(1, [[0, 0, 5, 5], [1, 1, 6, 6]]), record(2, [[2, 2, 7, 7]])]
    result = extraer.compare_identities(stored, same)
    assert result == {"coincide": True, "guardadas": 2, "nuevas": 2, "distintas": []}
    changed = [record(1, [[0, 0, 5, 5], [1, 1, 6, 6]]), record(2, [[2, 2, 7, 8]])]
    result = extraer.compare_identities(stored, changed)
    assert result["coincide"] is False and result["distintas"] == [2]
    result = extraer.compare_identities(stored, same[:1])
    assert result["coincide"] is False
    assert result["guardadas"] == 2 and result["nuevas"] == 1


def face(frame, box, vector):
    return {"frame": frame, "box": box, "embedding": np.array(vector, dtype=np.float32)}


def test_identity_arrays_round_trip(tmp_path):
    identities = [
        [face(0, [1, 2, 3, 4], [1, 0]), face(3, [5, 6, 7, 8], [0, 1])],
        [face(6, [9, 10, 11, 12], [0.6, 0.8])],
    ]
    arrays = extraer.identity_arrays(identities, [1, 2])
    doc = {"identities": [record(1, [[1, 2, 3, 4], [5, 6, 7, 8]]), record(2, [[9, 10, 11, 12]])]}
    out = str(tmp_path / "v")
    extraer.write_video(out, arrays, doc, {"video": "v"})
    assert os.path.exists(os.path.join(out, "extraccion.json"))
    items = entre_camaras.load_video_embeddings(out)
    assert [item["id"] for item in items] == [1, 2]
    assert items[0]["faces"].shape == (2, 2) and items[1]["faces"].shape == (1, 2)
    assert np.allclose(items[0]["best"], [0, 1])
    assert np.allclose(items[1]["best"], [0.6, 0.8])
    assert arrays["frame"].tolist() == [0, 3, 6]
    assert arrays["box"].tolist() == [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]
    assert arrays["identity"].tolist() == [1, 1, 2]
