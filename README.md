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

## Instalación

```bash
python3 -m venv venvs/eval
venvs/eval/bin/pip install -r requirements-eval.txt
venvs/eval/bin/python -m pytest -q tests
```

Los videos van en `videos/`. Los entornos `venvs/track` y `venvs/faces` son los del pipeline.

## Uso

Todos los comandos se ejecutan desde la raíz de este repositorio.

| Paso | Comando |
|---|---|
| Proponer rostros para etiquetar | `../tesis_face_recognition_pipeline/venvs/faces/bin/python -m harness.detect_frames --video videos/<archivo> --out ground_truth/<nombre> --pre-annotate` |
| Etiquetar | `venvs/eval/bin/python -m labeling.server --port 8765` y abrir el puerto 8765 desde VS Code |
| Ejecutar una prueba | `venvs/eval/bin/python -m harness.run configs/suites/<prueba>.yaml` |
| Enviar a etiquetar los rostros del pipeline | `venvs/eval/bin/python -m harness.gt03_export --all` |
| Recalcular tras etiquetar | `venvs/eval/bin/python -m harness.evaluate --all` |
| Generar el Excel | `venvs/eval/bin/python -m harness.export_excel` |

El Excel queda en `results/tuning_results.xlsx` y se regenera después de cada prueba.

## Contenido

| Ruta | Contenido | En git |
|---|---|---|
| `harness/` | ejecución de pruebas, métricas, registro y exportación a Excel | sí |
| `labeling/` | herramienta local de etiquetado | sí |
| `configs/` | parámetros base y una configuración por prueba | sí |
| `decisions.yaml` | decisiones tomadas a partir de los resultados | sí |
| `ground_truth/` | etiquetas (cajas e identificadores anónimos) | sí, sin imágenes |
| `videos/`, `runs/`, `results/` | videos, ejecuciones y resultados | no |

## Reglas

- Las pruebas nunca modifican el código del pipeline.
- Videos, fotogramas y recortes no salen de la máquina: muestran a personas reales.
