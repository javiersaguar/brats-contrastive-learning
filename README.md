# Aprendizaje contrastivo (SimCLR) para detección de gliomas en BraTS

Proyecto de investigación de **PASD** (Procesado Avanzado de Señales y Datos), ETSIT, Universidad Politécnica de Madrid.

**Autores:** Alejandro Cuevas · Iria Lozano · Javier Saguar

> **Resultado en una línea:** con solo 20 pacientes etiquetados, SimCLR obtiene AUC-ROC **0.9303** y supera a todos los métodos supervisados comparados. Con los 484 pacientes, la supervisión directa gana (CNN desde cero: 0.9798 frente a 0.8945 de SimCLR).

## Contenido del repositorio

| Ruta | Descripción |
|---|---|
| `notebooks/01_simclr_brats_pipeline.ipynb` | Carga de BraTS, preprocesado, preentrenamiento SimCLR, fine-tuning y evaluación |
| `notebooks/02_comparativa_supervisada.ipynb` | SimCLR frente a ResNet-50 supervisada, CNN desde cero, SVM y Random Forest |
| `figures/` | Curvas de pérdida, t-SNE (incluido 3D interactivo), matriz de confusión, ROC, ejemplos |
| `docs/blog_pasd.pdf` | Entrega final (formato blog de la asignatura) |
| `docs/informe_ssl_brats.pdf` | Informe del proyecto |

---

# Análisis del proyecto

*Basado en la entrega final (blog e informe en PDF) y en el documento Word de la propuesta inicial.*

## 1. Planteamiento y motivación

El diagnóstico de gliomas en resonancia magnética exige interpretar varias secuencias a la vez, y el etiquetado por radiólogos es caro y escaso. La hipótesis del trabajo es que el aprendizaje contrastivo puede aprender representaciones útiles **sin etiquetas** y transferirlas después a clasificación con muy pocos pacientes etiquetados. La pregunta que se contesta es *cuándo* compensa usarlo frente a un modelo supervisado.

## 2. Datos

- **BraTS** (Medical Segmentation Decathlon): 484 pacientes con glioma, volúmenes NIfTI de 240×240×155 vóxeles.
- Cuatro modalidades por paciente: FLAIR (edema), T1w (anatomía), T1gd (tumor activo con gadolinio) y T2w (agua, edema y necrosis).
- Tras el preprocesado quedan **66.517 cortes axiales 2D** válidos, casi equilibrados: 50.7 % con tumor y 49.3 % sin tumor.
- Preprocesado por canal: clipping al percentil 99.5, normalización z-score y redimensionado a 128×128.

## 3. Método

**Etapa 1: preentrenamiento contrastivo (sin etiquetas)**
- Encoder ResNet-50 inicializada con ImageNet y adaptada a 4 canales. El canal extra se inicializa con la media de los tres originales.
- Projection head MLP 2048→512→128 y pérdida **NT-Xent** (τ = 0.5).
- Dos vistas aumentadas por corte: flips, rotación ±15°, recorte aleatorio y blur gaussiano.
- 200 épocas, Adam (lr 1e-3, weight decay 1e-4), scheduler coseno.

**Etapa 2: fine-tuning para clasificación tumor / no tumor**
- Cabeza 2048→256→2 con BatchNorm, ReLU y Dropout 0.3.
- Fase 1: encoder congelado, 80 épocas. Fase 2: se descongelan `layer3` y `layer4`, otras 80 épocas.

**Corrección metodológica clave.** La primera versión hacía el split aleatorio por cortes, de modo que cortes del mismo paciente caían en train y validación (*data leakage*). Se rehízo con split **por paciente** (387 train / 97 validación). El AUC del prototipo bajó de 0.99 a 0.92, una cifra más honesta.

## 4. Resultados

### Clasificador final (484 pacientes, sin leakage)

| Métrica | Valor |
|---|---|
| AUC-ROC | 0.8945 |
| F1-Score | 0.8168 |
| Accuracy | 81 % |
| Validación estadística (5 runs, pacientes aleatorios) | AUC 0.9068 ± 0.0276 |
| Evaluación | 13.285 cortes de 97 pacientes no vistos |

### Efecto del preentrenamiento contrastivo

- Pérdida NT-Xent: train 2.8755, val 3.3574. La brecha de 0.48 es coherente con un split por paciente.
- Coeficiente de silueta del espacio latente: ResNet-50 sin adaptar **0.20** frente a encoder SimCLR **0.22**. La mejora es modesta. El tumor/no tumor se separa como propiedad emergente y no como objetivo del entrenamiento.

### Comparativa con métodos supervisados (AUC-ROC)

| Método | 20 pacientes | 484 pacientes | Etiquetas en preentrenamiento | Tiempo |
|---|---|---|---|---|
| **SimCLR (nuestro)** | **0.9303** | 0.8945 | Ninguna (solo fine-tuning) | ~600 min |
| ResNet-50 supervisada | 0.9195 | 0.9725 | Todas | ~57 min |
| CNN desde cero | 0.9277 | **0.9798** | Todas | ~56 min |
| SVM + ResNet | 0.8828 | 0.9431 | Todas | ~25 min |
| Random Forest + ResNet | 0.8645 | 0.9333 | Todas | ~1 min |

## 5. Hallazgos

1. **SimCLR gana con pocos datos.** Con 20 pacientes etiquetados supera a todos los supervisados. Es el caso clínico realista, donde etiquetar volúmenes cuesta horas de radiólogo.
2. **Con datos abundantes, la supervisión directa gana.** Con 484 pacientes, la CNN desde cero (0.98) supera claramente a SimCLR (0.89). El trabajo define el escenario óptimo de cada enfoque en lugar de afirmar una superioridad general.
3. **La CNN sin preentrenar supera a la ResNet-50 supervisada.** Una explicación posible, no demostrada, es que los sesgos de ImageNet dificultan adaptarse al dominio médico.
4. **Filtrar cortes "informativos" empeora el modelo.** Usar solo los cortes centrales (Z = 60–100) baja el AUC de 0.89 a 0.78, porque el desbalance resultante es extremo (86 % con tumor). La heterogeneidad del dataset completo es necesaria para generalizar.
5. **El split por paciente es imprescindible.** Evita estimaciones infladas por leakage.

## 6. Limitaciones y puntos a vigilar

- **Coste computacional:** SimCLR tarda ~600 min frente a ~1–57 min de los demás métodos. La ventaja con pocos datos tiene un precio alto en cómputo.
- **Silueta baja:** 0.17–0.22 indica una estructura latente moderada, no clusters nítidamente separados.
- **Una sola partición de validación** (97 pacientes) y un único dataset. Las notas de la entrega ya apuntan a que un split 70/30 sería más interesante, y justifican BraTS como dataset estándar y completo.
- **Clasificación a nivel de corte 2D.** No se aprovecha el contexto 3D del volumen.
- **Incoherencia menor en la entrega:** el blog da AUC 0.9246 para "20 pacientes" en el texto y 0.9303 en la tabla comparativa. Parece corresponder a ejecuciones distintas y conviene unificarlo.
- **Hiperparámetros:** el informe indica lr 3e-4 en la fase 2 del fine-tuning y el blog 1e-4. Hay que verificar cuál usa el notebook.

## 7. Evolución respecto a la propuesta inicial

La propuesta (documento Word) era más ambiciosa:
- Preentrenar con **OrganMNIST3D** (MedMNIST) y usar BraTS vía MONAI solo para la fase de transferencia.
- Enfoque **multitarea**: segmentación de lesiones con decodificador, clasificación, y pérdida combinada L = λ₁·L_contrastiva + λ₂·L_segmentación + λ₃·L_clasificación.
- Métricas AUC-ROC y Dice, y posible referencia a Alzheimer (ADNI).

La entrega final **acotó el alcance**: preentrenamiento directamente sobre BraTS, solo clasificación y métricas de AUC-ROC, F1 y silueta. La segmentación (cabeza U-Net, Dice) queda explícitamente como trabajo futuro.

## 8. Trabajo futuro

- VICReg o BYOL como alternativas más estables, sin pares negativos.
- Vision Transformers (ViT) para capturar dependencias globales.
- Aumentaciones específicas para resonancia magnética.
- Segmentación con cabeza U-Net sobre el encoder preentrenado y evaluación con Dice.
- Validación más robusta (validación cruzada por paciente o split 70/30).

## 9. Referencias principales

SimCLR (Chen et al., ICML 2020) · MoCo (He et al., CVPR 2020) · VICReg (Bardes et al., ICLR 2022) · BraTS (Menze et al., 2015; Bakas et al., 2017) · ResNet (He et al., 2016) · ViT (Dosovitskiy et al., 2021) · Azizi et al. (ICCV 2021) · Chaitanya et al. (NeurIPS 2020) · t-SNE (van der Maaten & Hinton, 2008). Lista completa con enlaces en `docs/blog_pasd.pdf`.

## Notas

- El dataset BraTS no se incluye; se descarga desde <http://medicaldecathlon.com/>.
- Los pesos entrenados (`*.pt`, ~100 MB cada uno) no se suben al repositorio.
