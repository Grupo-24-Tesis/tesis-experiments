import base64
import os
import sys

import numpy as np

PIPELINE_DIR = os.path.abspath(os.environ.get(
    "TESIS_PIPELINE_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                 "tesis_face_recognition_pipeline")))
sys.path.insert(0, PIPELINE_DIR)

import face_embedding


def test_identity_embedding_is_unit_mean():
    rng = np.random.default_rng(3)
    vectors = [face_embedding.unit_embedding(rng.normal(size=512)) for _ in range(5)]
    faces = [{"embedding": v} for v in vectors]

    mean = face_embedding.identity_embedding(faces)

    expected = np.sum(vectors, axis=0)
    expected = expected / np.linalg.norm(expected)
    assert mean.dtype == np.float32
    assert mean.shape == (512,)
    assert np.max(np.abs(mean - expected)) < 1e-6
    assert abs(np.linalg.norm(mean) - 1) < 1e-5


def test_encoded_embedding_is_float32_little_endian_base64():
    rng = np.random.default_rng(4)
    mean = face_embedding.unit_embedding(rng.normal(size=512))

    text = face_embedding.encode_embedding(mean)

    raw = base64.b64decode(text)
    assert len(raw) == 512 * 4
    assert np.array_equal(np.frombuffer(raw, dtype="<f4"), mean)


def test_embeddings_record_covers_each_saved_identity():
    rng = np.random.default_rng(5)
    identities = [[{"embedding": face_embedding.unit_embedding(rng.normal(size=512))}]
                  for _ in range(3)]

    record = face_embedding.embeddings_record("GhostFaceNet", identities, {1, 3})

    assert record["model"] == "GhostFaceNet"
    assert record["dimensions"] == 512
    assert sorted(record["embeddings"]) == ["1", "3"]
