import numpy as np
import pytest
from sklearn.metrics import adjusted_rand_score

from harness.metrics import clustering, detection, verification


def test_iou_basic():
    assert detection.iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0
    assert detection.iou([0, 0, 10, 10], [10, 10, 20, 20]) == 0.0
    assert detection.iou([0, 0, 10, 10], [5, 0, 15, 10]) == pytest.approx(50 / 150)


def test_match_tp_fp_fn():
    gts = [{"box": [0, 0, 20, 20], "identity": "P01"}, {"box": [100, 100, 120, 120], "identity": "P02"}]
    dets = [{"box": [1, 1, 21, 21], "score": 0.99}, {"box": [50, 50, 70, 70], "score": 0.95}]
    rows = detection.match_frame(dets, gts)
    outcomes = sorted(r["outcome"] for r in rows)
    assert outcomes == ["FN", "FP", "TP"]
    m = detection.prf(rows)
    assert (m["tp"], m["fp"], m["fn"]) == (1, 1, 1)
    assert m["f1"] == pytest.approx(0.5)


def test_one_gt_matched_once():
    gts = [{"box": [0, 0, 20, 20]}]
    dets = [{"box": [0, 0, 20, 20], "score": 0.9}, {"box": [1, 0, 21, 20], "score": 0.8}]
    m = detection.prf(detection.match_frame(dets, gts))
    assert (m["tp"], m["fp"], m["fn"]) == (1, 1, 0)


def test_ignored_and_small_faces_are_dropped():
    gts = [{"box": [0, 0, 20, 20], "ignore": True},
           {"box": [100, 100, 108, 108]},
           {"box": [200, 200, 240, 240], "occlusion": "total"}]
    dets = [{"box": [0, 0, 20, 20], "score": 0.9}, {"box": [100, 100, 108, 108], "score": 0.9}]
    m = detection.prf(detection.match_frame(dets, gts, min_px=16))
    assert (m["tp"], m["fp"], m["fn"]) == (0, 0, 0)


def test_threshold_turns_tp_into_fn():
    gts = [{"box": [0, 0, 20, 20]}]
    rows = detection.match_frame([{"box": [0, 0, 20, 20], "score": 0.6}], gts)
    assert detection.prf(rows, 0.5)["tp"] == 1
    assert detection.prf(rows, 0.7)["fn"] == 1


def test_average_precision_perfect_and_half():
    gts = [{"box": [0, 0, 20, 20]}, {"box": [50, 50, 70, 70]}]
    perfect = detection.match_frame([{"box": g["box"], "score": 0.9} for g in gts], gts)
    assert detection.average_precision(perfect) == pytest.approx(1.0)
    rows = detection.match_frame([{"box": [200, 200, 220, 220], "score": 0.9},
                                  {"box": [0, 0, 20, 20], "score": 0.8}], gts)
    assert detection.average_precision(rows) == pytest.approx(0.25)


def test_pairs_respect_time_gap():
    gen, imp = verification.build_pairs(["A", "A", "A", "B"], [0.0, 1.0, 5.0, 0.0], min_gap_s=2.0)
    assert {tuple(p) for p in gen} == {(0, 2), (1, 2)}
    assert len(imp) == 3


def test_tar_at_far_hand_case():
    imp = np.arange(1, 101) / 100.0
    gen = np.array([0.005, 0.01, 0.5, 0.9])
    res = verification.tar_at_far(gen, imp, 0.01)
    assert res["tau"] == pytest.approx(0.01)
    assert res["far_actual"] == pytest.approx(0.01)
    assert res["tar"] == pytest.approx(0.5)


def test_tar_at_far_ties_do_not_exceed_far():
    imp = np.array([0.1, 0.1, 0.5, 0.6])
    res = verification.tar_at_far(np.array([0.05, 0.1]), imp, 0.25)
    assert res["far_actual"] <= 0.25
    assert res["tar"] == pytest.approx(0.5)


def test_perfect_separation():
    m = verification.summarize(np.array([0.1, 0.2]), np.array([0.8, 0.9, 1.0]))
    assert m["emb.auc"] == pytest.approx(1.0)
    assert m["emb.eer"] == pytest.approx(0.0)
    assert m["emb.tar@far0.01"] == pytest.approx(1.0)


def test_cosine_distance():
    emb = np.array([[1.0, 0.0], [0.0, 2.0], [3.0, 0.0]])
    d = verification.pair_distances(emb, np.array([[0, 1], [0, 2]]))
    assert d == pytest.approx([1.0, 0.0])


def test_purity_and_bcubed_hand_case():
    true = ["A", "A", "B", "B"]
    pred = [0, 0, 0, 1]
    assert clustering.purity(true, pred) == pytest.approx(0.75)
    p, r, _ = clustering.bcubed(true, pred)
    assert p == pytest.approx((2 / 3 + 2 / 3 + 1 / 3 + 1) / 4)
    assert r == pytest.approx((1 + 1 + 0.5 + 0.5) / 4)


def test_noise_is_singletons():
    m = clustering.summarize(["A", "A", "B"], [-1, -1, 0])
    assert m["clu.n_pred"] == 1
    assert m["clu.noise_rate"] == pytest.approx(2 / 3)
    assert m["clu.purity"] == pytest.approx(1.0)
    assert m["clu.inverse_purity"] == pytest.approx(2 / 3)


def test_ari_matches_sklearn():
    true = ["A", "A", "B", "B", "C"]
    pred = [1, 1, 2, 3, 3]
    assert clustering.summarize(true, pred)["clu.ari"] == pytest.approx(adjusted_rand_score(true, pred))


def test_cluster_errors_kinds():
    rows = clustering.cluster_errors(["A", "A", "B", "A"], [0, 0, 0, 1])
    kinds = sorted(r["kind"] for r in rows)
    assert kinds == ["merge", "pure", "split"]
