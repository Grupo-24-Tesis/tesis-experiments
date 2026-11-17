import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))

from bench import bench_compuerta

XML = """<sequence name="A">
<frame number="1">
<person id="0001"><leftEye x="10" y="20"/><rightEye x="20" y="20"/></person>
</frame>
</sequence>"""


def write(path, content):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(content)


def test_find_ground_truth_composite_map_loads_every_referenced_xml(tmp_path, monkeypatch):
    monkeypatch.setenv("TESIS_CHOKEPOINT_DIR", str(tmp_path))
    write(str(tmp_path / "groundtruth" / "A.xml"), XML)
    document = {"frames": [None, ["A", 1], ["sin_gt", -1]], "empty_labelled": True}
    write(str(tmp_path / "runs" / "gating" / "videos_prueba" / "largo.frames.json"),
          json.dumps(document))
    kind, gts, frame_map = bench_compuerta.find_ground_truth("largo")
    assert kind == bench_compuerta.KIND_CHOKEPOINT
    assert list(gts) == ["A"]
    assert gts["A"][1] == [("0001", (15.0, 20.0))]
    assert frame_map == document


def test_find_ground_truth_legacy_map_still_needs_its_own_xml(tmp_path, monkeypatch):
    monkeypatch.setenv("TESIS_CHOKEPOINT_DIR", str(tmp_path))
    write(str(tmp_path / "groundtruth" / "V.xml"), XML)
    write(str(tmp_path / "videos" / "V.frames.json"), json.dumps([1, 1]))
    kind, gt, frame_map = bench_compuerta.find_ground_truth("V")
    assert kind == bench_compuerta.KIND_CHOKEPOINT
    assert gt[1] == [("0001", (15.0, 20.0))]
    assert frame_map == [1, 1]


def test_find_ground_truth_unknown_video(tmp_path, monkeypatch):
    monkeypatch.setenv("TESIS_CHOKEPOINT_DIR", str(tmp_path))
    assert bench_compuerta.find_ground_truth("zzz")[0] is None


def test_composite_map_keeps_empty_frames_of_the_xml_as_labelled_empty(tmp_path, monkeypatch):
    monkeypatch.setenv("TESIS_CHOKEPOINT_DIR", str(tmp_path))
    write(str(tmp_path / "groundtruth" / "A.xml"),
          '<dataset name="A">\n <frame number="00000005"/>\n</dataset>')
    write(str(tmp_path / "runs" / "gating" / "videos_prueba" / "c.frames.json"),
          json.dumps({"frames": [["A", 5]], "empty_labelled": True}))
    _, gts, _ = bench_compuerta.find_ground_truth("c")
    assert gts["A"][5] == []
