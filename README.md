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
| Comprobar el entorno (ENV-01) | `venvs/eval/bin/python -m harness.run configs/suites/ENV-01.yaml` |
| Ejecutar una prueba | `venvs/eval/bin/python -m harness.run configs/suites/<prueba>.yaml` |
| Enviar a etiquetar los rostros del pipeline | `venvs/eval/bin/python -m harness.gt03_export --all` |
| Recalcular tras etiquetar | `venvs/eval/bin/python -m harness.evaluate --all` |
| Generar el Excel | `venvs/eval/bin/python -m harness.export_excel` |

El Excel queda en `tuning_results.xlsx`, en la raíz del repositorio, se regenera después de cada prueba y se
guarda en git junto con el código que lo generó.

## Contenido

| Ruta | Contenido | En git |
|---|---|---|
| `harness/` | ejecución de pruebas, métricas, registro y exportación a Excel | sí |
| `labeling/` | herramienta local de etiquetado | sí |
| `configs/` | parámetros base y una configuración por prueba | sí |
| `decisions.yaml` | decisiones tomadas a partir de los resultados | sí |
| `tuning_results.xlsx` | resultados de todas las ejecuciones | sí |
| `ground_truth/` | etiquetas (cajas e identificadores anónimos) | sí, sin imágenes |
| `TEST_PLAN.md`, `findings.yaml` | plan de pruebas y hallazgos (los hallazgos aparecen en el Excel) | no |
| `videos/`, `runs/`, `results/` | videos, ejecuciones, registro y reportes con miniaturas | no |

## Reglas

- Las pruebas nunca modifican el código del pipeline.
- Videos, fotogramas y recortes solo se guardan en las máquinas del proyecto y nunca en git:
  muestran a personas reales.
