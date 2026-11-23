#!/usr/bin/env bash
set -u
EXPERIMENTS=/data/TesisFaceRecognition/tesis_face_recognition_experiments
PIPELINE=${TESIS_PIPELINE_DIR:-/data/TesisFaceRecognition/tesis_face_recognition_pipeline}
CHOKEPOINT=${TESIS_CHOKEPOINT_DIR:-/data/TesisFaceRecognition/chokepoint}
OUT=${1:-$EXPERIMENTS/runs/flujo_actual}
EVAL=$EXPERIMENTS/venvs/eval/bin/python
export TESIS_PIPELINE_DIR=$PIPELINE TESIS_CHOKEPOINT_DIR=$CHOKEPOINT
mkdir -p "$OUT"
cd "$EXPERIMENTS"

step() {
  local name=$1
  shift
  if grep -q "^$name 0 " "$OUT/estado.txt" 2>/dev/null; then
    return
  fi
  local started
  started=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  "$@" > "$OUT/$name.log" 2>&1
  local code=$?
  echo "$name $code $started $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$OUT/estado.txt"
}

{
  echo "pipeline $(git -C "$PIPELINE" rev-parse HEAD) $(git -C "$PIPELINE" branch --show-current)"
  echo "pipeline_cambios $(git -C "$PIPELINE" status --porcelain | wc -l)"
  echo "experimentos $(git rev-parse HEAD) $(git branch --show-current)"
  echo "experimentos_cambios $(git status --porcelain | wc -l)"
} > "$OUT/versiones.txt"

CROSS=(P1E_S1_C1 P1E_S1_C2 P1E_S2_C1 P2E_S1_C2.1)
VIDEOS=("$EXPERIMENTS/videos/Camera_ch6_20261016100358_20261016100614.mp4" "$EXPERIMENTS/videos/Camera3.mp4")
for name in "${CROSS[@]}"; do
  VIDEOS+=("$CHOKEPOINT/videos/$name.mp4")
done
VIDEOS+=("$CHOKEPOINT/runs/gating/videos_prueba/borde_inicio.mp4" "$CHOKEPOINT/runs/gating/videos_prueba/borde_seguidos.mp4")

step pruebas "$EVAL" -m pytest -q tests
step entorno "$EVAL" -m harness.run configs/suites/ENV-01.yaml
step compuerta "$EVAL" -m bench.bench_compuerta run --label compuerta_defecto --out "$OUT" --gated "${VIDEOS[@]}"
step base "$EVAL" -m bench.bench_compuerta run --label sin_compuerta --out "$OUT" "${VIDEOS[@]}"
step comparacion "$EVAL" -m bench.bench_compuerta compare "$OUT/sin_compuerta/results.json" "$OUT/compuerta_defecto/results.json"
step extraccion "$EVAL" -m bench.entre_camaras_corrida --run "$OUT/compuerta_defecto" --out "$OUT/entre_camaras"
step entre_camaras "$EVAL" -m bench.entre_camaras --runs "$OUT/entre_camaras"
touch "$OUT/terminado"
