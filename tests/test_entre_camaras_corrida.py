import base64
import json
import os

import numpy as np

from bench import entre_camaras, entre_camaras_corrida as corrida


def encode(vector):
    return base64.b64encode(np.asarray(vector, dtype="<f4").tobytes()).decode()


def test_identity_arrays_index_best_by_identity():
    document = {"dimensions": 2, "embeddings": {"3": encode([0.0, 1.0]), "1": encode([1.0, 0.0])}}
    arrays = corrida.identity_arrays(document)
    assert arrays["identity"].tolist() == [1, 3]
    assert arrays["embeddings"].tolist() == [[1.0, 0.0], [0.0, 1.0]]
    assert arrays["best"][0] == 0 and arrays["best"][2] == 1


def test_identity_arrays_without_detections():
    arrays = corrida.identity_arrays({"dimensions": 4, "embeddings": {}})
    assert arrays["embeddings"].shape == (0, 4)
    assert len(arrays["identity"]) == 0


def test_convert_video_is_readable_by_scorer(tmp_path):
    source = tmp_path / "corrida" / "P1E_S1_C1"
    source.mkdir(parents=True)
    (source / "embeddings.json").write_text(json.dumps(
        {"dimensions": 2, "embeddings": {"2": encode([0.6, 0.8])}}))
    (source / "identities.json").write_text(json.dumps({"identities": [{"id": 2, "faces": []}]}))
    out = tmp_path / "salida" / "P1E_S1_C1"
    summary = corrida.convert_video(str(source), str(out), "P1E_S1_C1")
    assert summary["identidades"] == 1 and summary["coincide"]
    items = entre_camaras.load_video_embeddings(str(out))
    assert [item["id"] for item in items] == [2]
    assert np.allclose(items[0]["faces"], [[0.6, 0.8]])
    assert np.allclose(items[0]["best"], [0.6, 0.8])
    assert os.path.exists(out / entre_camaras.EXTRACTION_NAME)
