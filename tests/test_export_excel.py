import os

import pandas as pd

from harness import export_excel


def test_device_text_reads_devices_as_words():
    assert export_excel.device_text(["CUDAExecutionProvider", "CPUExecutionProvider"]) == \
        "GPU (CUDA), con CPU de respaldo"
    assert export_excel.device_text("CUDAExecutionProvider,CPUExecutionProvider") == \
        "GPU (CUDA), con CPU de respaldo"
    assert export_excel.device_text("cuda:0") == "GPU (cuda:0)"
    assert export_excel.device_text("/physical_device:GPU:0") == "GPU 0"
    assert export_excel.device_text("Tesla T4") == "Tesla T4"


def test_plain_text_and_shown_text():
    assert export_excel.plain_text({"a": [1, 2]}) == "a: 1, 2"
    assert export_excel.shown_text(445.3) == "445.3000"
    assert export_excel.shown_text(None) == ""


def test_env_detail_is_readable(tmp_path, monkeypatch):
    run_dir = tmp_path / "runs" / "ENV-01" / "r1"
    (run_dir / "classification").mkdir(parents=True)
    (run_dir / "classification" / "env_checks.csv").write_text(
        "check,venv,value,expected,passed\n"
        "GPU codificador ReID,track,\"CUDAExecutionProvider,CPUExecutionProvider\","
        "CUDAExecutionProvider primero,True\n"
        "sin archivos fuera del repositorio,,0,0,False\n")
    monkeypatch.setattr(export_excel.runlog, "EXP_DIR", str(tmp_path))
    rows = export_excel.env_detail(pd.DataFrame([{"run_dir": os.path.join("runs", "ENV-01", "r1")}]))
    assert rows.to_dict("records") == [
        {"comprobación": "GPU codificador ReID", "entorno": "track",
         "obtenido": "GPU (CUDA), con CPU de respaldo", "esperado": "GPU", "resultado": "aprobada"},
        {"comprobación": "sin archivos fuera del repositorio", "entorno": "todos",
         "obtenido": "0", "esperado": "0", "resultado": "no aprobada"},
    ]
