import os
import sys

import numpy as np
import pytest

PIPELINE_DIR = os.path.abspath(os.environ.get(
    "TESIS_PIPELINE_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                 "tesis_face_recognition_pipeline")))
sys.path.insert(0, PIPELINE_DIR)

import face_embedding

INPUT_SHAPE = (112, 112)


class FakeModel:
    def __init__(self, flat_single=False):
        self.batches = []
        self.flat_single = flat_single

    def forward(self, batch):
        self.batches.append(batch.copy())
        rows = np.stack([batch[:, 0, 0, 0] * 1000 + 1, batch[:, 0, 0, 1] * 1000 + 2,
                         batch[:, 0, 0, 2] * 1000 + 3], axis=1)
        if self.flat_single and len(rows) == 1:
            return rows[0].tolist()
        return rows.tolist()


def colour_face(b, g, r, size=112):
    face = np.zeros((size, size, 3), dtype=np.uint8)
    face[..., 0], face[..., 1], face[..., 2] = b, g, r
    return face


def test_aligned_face_keeps_size_dtype_and_bgr_order_scaled_to_one():
    face = colour_face(10, 128, 255)

    img = face_embedding.model_input(face, INPUT_SHAPE)

    assert img.shape == (112, 112, 3)
    assert img.dtype == np.float32
    assert img[0, 0].tolist() == pytest.approx([10 / 255, 128 / 255, 1.0])
    assert np.array_equal(img, face.astype(np.float32) / np.float32(255.0))


def test_image_already_in_unit_range_is_not_divided_again():
    face = colour_face(0, 1, 1)

    img = face_embedding.model_input(face, INPUT_SHAPE)

    assert img.max() == 1.0
    assert img[0, 0].tolist() == [0.0, 1.0, 1.0]


def test_non_square_face_is_scaled_and_padded_in_the_middle_like_deepface():
    face = colour_face(255, 255, 255, size=56)
    face = np.concatenate([face, face], axis=1)

    img = face_embedding.model_input(face, INPUT_SHAPE)

    assert img.shape == (112, 112, 3)
    assert img[:28].max() == 0.0
    assert img[84:].max() == 0.0
    assert img[28:84].min() == 1.0


def test_small_face_is_upscaled_to_the_model_size():
    face = colour_face(255, 0, 0, size=56)

    img = face_embedding.model_input(face, INPUT_SHAPE)

    assert img.shape == (112, 112, 3)
    assert img[..., 0].min() == 1.0
    assert img[..., 1:].max() == 0.0


def test_batches_have_a_fixed_size_padded_with_zeros_and_keep_the_order():
    model = FakeModel()
    faces = [colour_face(k + 2, 0, 0) for k in range(70)]

    embeddings = face_embedding.embed_faces(model.forward, faces, INPUT_SHAPE)

    assert [b.shape for b in model.batches] == [(64, 112, 112, 3), (64, 112, 112, 3)]
    assert model.batches[1][6:].max() == 0.0
    assert len(embeddings) == 70
    raw = np.array([[(k + 2) / 255 * 1000 + 1, 2, 3] for k in range(70)], dtype=np.float32)
    expected = raw / np.linalg.norm(raw, axis=1, keepdims=True)
    assert np.abs(np.array(embeddings) - expected).max() < 1e-6


def test_embeddings_have_unit_norm_and_match_the_per_face_normalisation():
    model = FakeModel()
    faces = [colour_face(30, 60, 90), colour_face(200, 10, 5)]

    embeddings = face_embedding.embed_faces(model.forward, faces, INPUT_SHAPE, chunk=4)

    for face, emb in zip(faces, embeddings):
        single = FakeModel().forward(face_embedding.model_input(face, INPUT_SHAPE)[None])[0]
        assert np.linalg.norm(emb) == pytest.approx(1.0, abs=1e-6)
        assert np.array_equal(emb, face_embedding.unit_embedding(single))


def test_single_face_batches_accept_the_flat_output_of_deepface():
    model = FakeModel(flat_single=True)

    embeddings = face_embedding.embed_faces(model.forward, [colour_face(0, 0, 255)],
                                            INPUT_SHAPE, chunk=1)

    assert [b.shape for b in model.batches] == [(1, 112, 112, 3)]
    assert len(embeddings) == 1
    assert embeddings[0].shape == (3,)


def test_no_faces_means_no_model_call():
    model = FakeModel()

    assert face_embedding.embed_faces(model.forward, [], INPUT_SHAPE) == []
    assert model.batches == []


def test_unit_embedding_matches_the_previous_per_face_formula():
    values = [3.0, 4.0, 0.0]

    emb = face_embedding.unit_embedding(values)

    assert emb.dtype == np.float32
    assert emb.tolist() == pytest.approx([0.6, 0.8, 0.0])
