# tesis-experiments

Pruebas y ajuste de parámetros de [tesis_face_recognition_pipeline](https://github.com/Grupo-24-Tesis/tesis_face_recognition_pipeline). Cada ejecución deja su propio directorio con configuración, registros y métricas, y todos los resultados se reúnen en un libro de Excel.

## Estructura

Este repositorio debe estar junto al del pipeline:

```
/data/TesisFaceRecognition/
  tesis_face_recognition_pipeline/      código de producción, venvs/track, venvs/faces, pesos de los modelos
  tesis_face_recognition_experiments/   este repositorio, venvs/eval
```

Para usar otra copia del pipeline, definir `TESIS_PIPELINE_DIR` con su ruta.

En Ubuntu 26.04, el driver de NVIDIA y Python 3.12 se instalan como indica el README del pipeline.

## Instalación

```bash
python3.12 -m venv venvs/eval
venvs/eval/bin/pip install -r requirements-eval.txt
venvs/eval/bin/python -m pytest -q tests
```

Los videos van en `videos/`. Los entornos `venvs/track`, `venvs/faces` y `venvs/worker` son los del
pipeline. `requirements-eval.txt` fija numpy 2.3.5: desde numpy 2.4 la agrupación HDBSCAN de
scikit-learn 1.9.1 falla cuando `cluster_selection_epsilon` es mayor que 0 (`TypeError: only
0-dimensional arrays can be converted to Python scalars`).

## Uso

Todos los comandos se ejecutan desde la raíz de este repositorio.

| Paso | Comando |
|---|---|
| Proponer rostros para etiquetar | `../tesis_face_recognition_pipeline/venvs/faces/bin/python -m harness.detect_frames --video videos/<archivo> --out ground_truth/<nombre> --pre-annotate` |
| Etiquetar | `venvs/eval/bin/python -m labeling.server --port 8765` y abrir el puerto 8765 desde VS Code |
| Probar el flujo actual | `bench/flujo_actual.sh` (escribe en `runs/flujo_actual/`; al repetirlo omite los pasos que ya terminaron con código 0) |
| Comprobar el entorno (ENV-01) | `venvs/eval/bin/python -m harness.run configs/suites/ENV-01.yaml` |
| Ejecutar una prueba del flujo anterior | `venvs/eval/bin/python -m harness.run configs/suites/<prueba>.yaml` |
| Enviar a etiquetar los rostros del pipeline | `venvs/eval/bin/python -m harness.gt03_export --all` |
| Recalcular tras etiquetar | `venvs/eval/bin/python -m harness.evaluate --all` |
| Generar el Excel | `venvs/eval/bin/python -m harness.export_excel` |

Las pruebas de `harness.run`, salvo ENV-01, miden el flujo anterior basado en seguimiento de personas y
sirven para reproducir los experimentos previos. El Excel queda en `tuning_results.xlsx`, en la raíz del
repositorio, guarda solo los resultados del flujo actual y se versiona en git junto con el código que lo generó.

## Contenido

| Ruta | Contenido | En git |
|---|---|---|
| `harness/` | ejecución de pruebas, métricas, registro y exportación a Excel | sí |
| `bench/` | pruebas de rendimiento y precisión del flujo actual | sí |
| `labeling/` | herramienta local de etiquetado | sí |
| `configs/` | parámetros base y una configuración por prueba | sí |
| `decisions.yaml` | decisiones tomadas a partir de los resultados | sí |
| `tuning_results.xlsx` | resultados del flujo actual | sí |
| `ground_truth/` | etiquetas (cajas e identificadores anónimos) | sí, sin imágenes |
| `TEST_PLAN.md`, `findings.yaml` | plan de pruebas y hallazgos (los hallazgos aparecen en el Excel) | no |
| `videos/`, `runs/`, `results/` | videos, ejecuciones, registro y reportes con miniaturas | no |

## Reglas

- Las pruebas nunca modifican el código del pipeline.
- Videos, fotogramas y recortes solo se guardan en las máquinas del proyecto y nunca en git:
  muestran a personas reales.

## Pruebas del flujo actual

`bench/flujo_actual.sh` comprueba que el flujo de producción (compuerta de presencia y `face_identities.py`)
detecta los rostros, agrupa a cada persona en una sola identidad por video y permite encontrarla en otras
cámaras. Ejecuta el pipeline sin cambios, con los parámetros de producción, y corre en orden:

| Paso | Qué hace |
|---|---|
| PRU-01 | pytest de este repositorio: herramientas de métricas y `worker.py` |
| ENV-01 | bibliotecas, GPU y lectura de video en `venvs/track`, `faces`, `eval` y `worker` |
| FLU-01 | `bench.bench_compuerta run --gated`: el flujo de producción |
| FLU-02 | la misma corrida sin `--gated` y `bench.bench_compuerta compare`: compuerta frente a sin compuerta |
| FLU-03 | `bench.entre_camaras_corrida` toma el embedding medio por identidad que envía el worker y `bench.entre_camaras` lo puntúa |

Videos, reducidos para que la prueba dure una hora (queda fuera el de 45 minutos): Camera_ch6 y Camera3
(residenciales, etiquetas de `ground_truth/`), los ChokePoint P1E_S1_C1, P1E_S1_C2, P1E_S2_C1 y P2E_S1_C2.1
(verdad de terreno oficial) y los clips de borde de la compuerta borde_inicio y borde_seguidos. Se miden
precisión y cobertura de la detección, personas encontradas, duplicados, identidades mezcladas y pureza,
detecciones perdidas por la compuerta y factor de tiempo, y hit@k, precision@10 y precisión y cobertura por
umbral de similitud.

Corrida del 2026-11-22 (Tesla T4, pipeline `a9c95c7`): pasan las 459 pruebas de PRU-01 y las 14
comprobaciones de ENV-01. En ChokePoint la detección tiene 97.0 % de precisión y 99.9 % de cobertura, con 95
de 99 personas encontradas, 0 duplicados y 6 identidades mezcladas; en los videos residenciales, 85.8 % y
80.1 %, con 9 de 17 personas. La compuerta no pierde ninguna detección; en estos videos ocupados cuesta algo
de tiempo (factor 1.74 frente a 1.68) y ahorra en videos casi vacíos. Entre cámaras, con las 99 identidades
de los 4 videos ChokePoint, el primer candidato es correcto en el 100 % de las búsquedas (azar 3.8 %), y con
umbral 0.55 la precisión es 98.0 % y la cobertura 100 %. El detalle está en las hojas Resumen, Flujo -
totales, Flujo - videos, Entre cámaras y Entre cámaras - umbrales del Excel.

La medición entre cámaras más amplia, sobre los 72 videos de ChokePoint con verdad de terreno, se repite con
la extracción completa, que guarda un embedding por rostro, y su puntuación:

```
cd /data/TesisFaceRecognition/tesis_face_recognition_experiments && /data/TesisFaceRecognition/tesis_face_recognition_pipeline/venvs/faces/bin/python -m bench.entre_camaras_extraer
cd /data/TesisFaceRecognition/tesis_face_recognition_experiments && venvs/eval/bin/python -m bench.entre_camaras
```

Los resultados quedan en `/data/TesisFaceRecognition/chokepoint/runs/entre_camaras/` (una carpeta por video y `resultados.json`). Los embeddings nunca se guardan en git. Una extracción interrumpida se retoma sola: se omiten los videos que ya tienen `extraccion.json`.
