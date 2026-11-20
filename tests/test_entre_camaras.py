import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))

import numpy as np
import pytest

from bench import entre_camaras


def test_parse_video_name():
    assert entre_camaras.parse_video_name("P2L_S3_C2.2") == {
        "portal": 2, "direction": "L", "session": 3, "camera": 2, "part": 2}
    assert entre_camaras.parse_video_name("P1E_S1_C1")["part"] is None
    with pytest.raises(ValueError, match="nombre de video"):
        entre_camaras.parse_video_name("Camera3")


def test_relation():
    relation = entre_camaras.relation
    assert relation("P1E_S1_C1", "P1E_S1_C2") == "misma_pasada"
    assert relation("P1E_S1_C1", "P1E_S2_C1") == "otra_pasada"
    assert relation("P1E_S1_C1", "P1L_S1_C1") == "otra_pasada"
    assert relation("P2E_S1_C1.1", "P2E_S1_C3.1") == "misma_pasada"
    assert relation("P2E_S1_C1.1", "P2E_S1_C2.2") == "otra_pasada"
    assert relation("P1E_S1_C1", "P2E_S1_C1.1") == "otro_portal"
    assert relation("P1E_S1_C1", "P1E_S1_C1") is None


def test_person_key():
    assert entre_camaras.person_key("P1E_S1_C1:0005") == "0005"
    assert entre_camaras.person_key("0005") == "0005"


def test_identity_labels():
    gt = {10: [("0005", (105, 105))], 20: [("0005", (105, 105)), ("0007", (205, 205))]}
    identities = [
        {"id": 1, "faces": [{"frame": 10, "box": [100, 100, 110, 110]},
                            {"frame": 20, "box": [100, 100, 110, 110]}]},
        {"id": 2, "faces": [{"frame": 20, "box": [200, 200, 210, 210]},
                            {"frame": 20, "box": [100, 100, 110, 110]}]},
        {"id": 3, "faces": [{"frame": 30, "box": [100, 100, 110, 110]}]},
    ]
    assert entre_camaras.identity_labels(identities, gt, None) == {1: "0005", 2: None, 3: None}


def make_query_and_gallery():
    query = {"faces": np.array([[1, 0], [0, 1]], dtype=np.float32),
             "best": np.array([1, 0], dtype=np.float32)}
    items = [{"faces": np.array([[1, 0]], dtype=np.float32), "best": np.array([1, 0], dtype=np.float32)},
             {"faces": np.array([[0.6, 0.8]], dtype=np.float32),
              "best": np.array([0.6, 0.8], dtype=np.float32)}]
    return query, items


def test_query_scores_variants():
    query, items = make_query_and_gallery()
    gallery = entre_camaras.build_gallery(items)
    expected = {"media": [0.7071, 0.9899], "mejor": [1.0, 0.6], "par_maximo": [1.0, 0.8]}
    for variant, values in expected.items():
        scores = entre_camaras.query_scores(gallery, query, variant)
        assert np.allclose(scores, values, atol=1e-4)


QUERIES = [(np.array([0.9, 0.8, 0.1]), np.array([False, True, False])),
           (np.array([0.5, 0.4]), np.array([True, False]))]


def test_search_metrics():
    result = entre_camaras.search_metrics(QUERIES)
    assert result["consultas"] == 2
    assert result["acierto_1"] == pytest.approx(0.5)
    assert result["acierto_5"] == pytest.approx(1.0)
    assert result["rango_medio"] == pytest.approx(1.5)
    assert result["ap_media"] == pytest.approx(0.75)
    assert result["precision_10"] == pytest.approx(0.1)
    assert result["precision_10_max"] == pytest.approx(0.1)
    assert result["azar"]["acierto_1"] == pytest.approx(0.41667, abs=1e-4)
    assert result["azar"]["rango_medio"] == pytest.approx(1.75)
    assert result["azar"]["acierto_5"] == pytest.approx(1.0)
    assert result["azar"]["precision_10"] == pytest.approx(0.1)


def test_ties_rank_pessimistically():
    result = entre_camaras.search_metrics([(np.array([0.5, 0.5]), np.array([True, False]))])
    assert result["rango_medio"] == pytest.approx(2.0)
    assert result["acierto_1"] == pytest.approx(0.0)


def test_threshold_table():
    table = entre_camaras.threshold_table(QUERIES, [0.0, 0.45, 0.85])
    assert [row["precision"] for row in table] == pytest.approx([0.4, 0.6667, 0.0], abs=1e-4)
    assert [row["exhaustividad"] for row in table] == pytest.approx([1.0, 1.0, 0.0])
    assert [row["resultados_por_consulta"] for row in table] == pytest.approx([2.5, 1.5, 0.5])
    assert entre_camaras.threshold_table(QUERIES, [0.95])[0]["precision"] is None


def make_items():
    def item(video, identity, label, vector):
        face = np.array([vector], dtype=np.float32)
        return {"video": video, "id": identity, "label": label, "faces": face, "best": face[0]}

    return [item("P1E_S1_C1", 1, "0001", [1, 0]), item("P1E_S1_C1", 2, None, [0, 1]),
            item("P1E_S1_C2", 1, "0001", [0.6, 0.8]), item("P2E_S1_C1.1", 1, "0002", [0, 1])]


def test_evaluate_cases():
    result = entre_camaras.evaluate(make_items())
    for variant in entre_camaras.VARIANTS:
        assert result[variant]["todas"]["metricas"]["consultas"] == 2
        assert result[variant]["todas"]["sin_objetivo"] == 1
        assert result[variant]["misma_pasada"]["metricas"]["consultas"] == 2
        assert result[variant]["misma_pasada"]["sin_objetivo"] == 1


def test_evaluate_empty_case_has_none_metrics():
    result = entre_camaras.evaluate(make_items())
    for variant in entre_camaras.VARIANTS:
        case = result[variant]["otro_portal"]
        assert case["metricas"]["consultas"] == 0
        assert case["metricas"]["acierto_1"] is None
        assert case["sin_objetivo"] == 3


def test_load_video_embeddings(tmp_path):
    rows = np.array([[1, 0], [0, 1], [0.6, 0.8]], dtype=np.float32)
    np.savez(tmp_path / "embeddings.npz", embeddings=rows, identity=np.array([1, 1, 2]),
             frame=np.array([0, 1, 2]), box=np.zeros((3, 4), dtype=int), best=np.array([1, 2]))
    items = entre_camaras.load_video_embeddings(str(tmp_path))
    assert [item["id"] for item in items] == [1, 2]
    assert items[0]["faces"].shape == (2, 2)
    assert items[1]["faces"].shape == (1, 2)
    assert np.allclose(items[0]["best"], rows[1])
    assert np.allclose(items[1]["best"], rows[2])
