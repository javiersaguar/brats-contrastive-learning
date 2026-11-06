# Aprendizaje contrastivo (SimCLR) para detección de tumores cerebrales en BraTS

Proyecto de investigación de PASD (Universidad Politécnica de Madrid).

**Autores:** Alejandro Cuevas · Iria Lozano · Javier Saguar

Pipeline de aprendizaje contrastivo autosupervisado (SimCLR) sobre resonancia magnética cerebral multimodal (dataset BraTS) y comparación con métodos supervisados.

## Contenido
- `notebooks/01_simclr_brats_pipeline.ipynb` — carga de BraTS, preentrenamiento contrastivo, fine-tuning y evaluación.
- `notebooks/02_comparativa_supervisada.ipynb` — SimCLR frente a métodos supervisados.
- `figures/` — curvas de pérdida, t-SNE, matriz de confusión, ROC, etc.
- `docs/informe_ssl_brats.pdf` — informe.

## Notas
- El dataset BraTS no se incluye; hay que descargarlo por separado.
- Los pesos entrenados (`*.pt`, ~100 MB cada uno) no se suben al repositorio.
