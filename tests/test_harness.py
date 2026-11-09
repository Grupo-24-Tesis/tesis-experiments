import numpy as np

from harness import cluster, registry, run
from labeling.server import validate_frame

BASE = {"video": "v.mp4", "stage1": {"A": 1, "B": 2}, "cluster": {"algo": "hdbscan", "dbscan": {"eps": 0.5}}}


def test_expand_dedupes_baseline_and_applies_groups():
    suite = {"points": [{"label": "baseline"}],
             "one_at_a_time": {"stage1.A": [1, 3]},
             "groups": [{"label": "dbscan", "set": {"cluster.algo": "dbscan"},
                         "grid": {"cluster.dbscan.eps": [0.3, 0.4]}}]}
    points = run.expand(suite, BASE)
    labels = [p[0] for p in points]
    assert labels == ["baseline", "stage1.A=3", "dbscan eps=0.3", "dbscan eps=0.4"]
    assert points[2][2]["cluster"]["algo"] == "dbscan"
    assert BASE["stage1"]["A"] == 1


def test_distance_matrix_cosine():
    d = cluster.distance_matrix(np.array([[1.0, 0.0], [2.0, 0.0], [0.0, 1.0]]))
    assert np.allclose(d, [[0, 0, 1], [0, 0, 1], [1, 1, 0]])


def test_clusters_and_cannot_link():
    emb = np.array([[1.0, 0.0], [0.99, 0.05], [0.0, 1.0], [0.05, 0.99]])
    d = cluster.distance_matrix(emb)
    labels = cluster.cluster(d, "agglomerative", {"linkage": "average", "distance_threshold": 0.3})
    assert labels[0] == labels[1] and labels[2] == labels[3] and labels[0] != labels[2]
    split = cluster.cluster(cluster.apply_cannot_link(d, [(0, 1)]), "agglomerative",
                            {"linkage": "average", "distance_threshold": 0.3})
    assert split[0] != split[1]


def test_dbscan_noise():
    emb = np.array([[1.0, 0.0], [0.99, 0.05], [0.0, 1.0]])
    labels = cluster.cluster(cluster.distance_matrix(emb), "dbscan", {"eps": 0.1, "min_samples": 2})
    assert labels[2] == -1 and labels[0] == labels[1] != -1


def test_hdbscan_with_selection_epsilon_merges_nested_clusters():
    rng = np.random.default_rng(0)
    x = np.concatenate([np.array(c) + rng.normal(scale=0.01, size=(10, 2)) for c in ([0, 0], [0.2, 0], [10, 0])])
    d = np.linalg.norm(x[:, None] - x[None], axis=-1)
    labels = cluster.cluster(d, "hdbscan", {"min_cluster_size": 2, "min_samples": 1,
                                            "cluster_selection_epsilon": 0.5})
    assert len(set(labels[:20])) == 1 and len(set(labels[20:])) == 1 and labels[0] != labels[20]


def test_verdict_fails_when_the_output_prevents_the_metric():
    targets = {"xv.open_set_accuracy": 0.9}
    assert registry.verdict({"metric.xv.open_set_accuracy": None}, targets) == "PENDING"
    assert registry.verdict({"metric.xv.open_set_accuracy": None, "unmet": "sin personas"}, targets) == "FAIL"
    assert registry.verdict({"metric.xv.open_set_accuracy": 0.95}, targets) == "PASS"


def test_rewritten_pipeline_commit_is_shown_with_its_new_hash():
    cols = registry.git_columns({"pipeline": {"commit": "ff1511563d1636f272304d31adbba9061c883378", "dirty": True},
                                 "experiments": {"commit": "HEAD", "dirty": False}})
    assert cols["git_pipeline_commit"] == "1e38058b31"
    assert cols["git_experiments_commit"] == "sin commits"


def test_validate_frame_requires_fields_unless_ignored():
    ok = {"faces": [{"box": [0, 0, 5, 5], "identity": "P01", "pose": "frontal", "occlusion": "none"},
                    {"box": [0, 0, 5, 5], "ignore": True}]}
    assert validate_frame(ok) == []
    bad = {"faces": [{"box": [0, 0, 5, 5], "identity": None, "pose": "x", "occlusion": None}]}
    assert len(validate_frame(bad)) == 3


def test_evaluate_failure_marks_run_invalid(tmp_path, monkeypatch):
    from harness import evaluate, registry, runlog
    run_dir = tmp_path / "CLU-01" / "r1"
    run_dir.mkdir(parents=True)
    runlog.write_json(str(run_dir / "run.json"), {"run_id": "r1", "test_id": "CLU-01", "kind": "cluster", "status": "ok"})
    runlog.write_json(str(run_dir / "config.json"), {"video": "videos/x.mp4"})

    def boom(*args):
        raise TypeError("only 0-dimensional arrays can be converted to Python scalars")
    monkeypatch.setitem(evaluate.EVALUATORS, "cluster", boom)
    monkeypatch.setattr(evaluate, "load_gt", lambda v: {})
    m = evaluate.evaluate_run(str(run_dir))
    assert m["_error"].startswith("TypeError")
    assert any("falló la evaluación" in p for p in registry.validate(str(run_dir)))
    assert '"event": "error"' in (run_dir / "events.jsonl").read_text()


def test_group_people_joins_tracks_by_majority_vote():
    from harness import evaluate
    its = [{"track_id": t} for t in ("1", "1", "1", "2", "2", "3")]
    labels = np.array([0, 0, 1, 0, 1, 1])
    tracks = {t: {} for t in ("1", "2", "3", "4")}
    groups = evaluate.group_people(its, labels, tracks)
    assert sorted(map(sorted, groups)) == [["1", "2"], ["3"], ["4"]]


def unit(v):
    v = np.asarray(v, dtype=float)
    return (v / np.linalg.norm(v)).tolist()


def test_xv_open_set_outcomes():
    from harness import evaluate
    def person(n, pid, v, unsure=False):
        return {"person": n, "identity": pid, "centroid": unit(v), "uncertain": unsure}
    gal = {"video": "a", "camera": "c6", "people": [person(0, "P01", [1, 0, 0]), person(1, "P02", [0, 1, 0])]}
    probe = {"video": "b", "camera": "c6", "people": [
        person(0, "P01", [1, 0.1, 0]), person(1, "P09", [0, 0, 1]), person(2, "P02", [0.9, 0.3, 0]),
        person(3, "P10", [0.95, 0.05, 0], unsure=True)]}
    other = {"video": "c", "camera": "c3", "people": [person(0, "C3-01", [1, 0, 0.05])]}
    rows = [r for r in evaluate.xv_open_set([gal, probe, other], 0.3) if r["gallery_video"] == "a"]
    outcome = {r["identity"]: r["outcome"] for r in rows}
    assert outcome == {"P01": "correct_match", "P09": "correct_reject", "P02": "wrong_match",
                       "P10": "unverified_match", "C3-01": "unverified_match"}
    rates = evaluate.xv_rates(rows, "")
    assert rates["n_probes"] == 3 and rates["unverified_matches"] == 2
    assert rates["accuracy"] == 2 / 3 and rates["wrong_match_rate"] == 1 / 3
