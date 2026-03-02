# Proyecto Kaggle: Freesound Audio Tagging

Clasificación de audio multiclase usando deep learning para la competencia de Freesound en Kaggle.

## Estructura del Proyecto

### Notebooks de Modelos Finales (`notebooks/modelos-finales/`)

- **EDA.ipynb** - Análisis exploratorio de datos de audio
- **procesamiento_de_audios.ipynb** - Pipeline de procesamiento y feature extraction
- **best_mobilenet.ipynb** - Modelo final MobileNet con mejores resultados

### Notebooks de Exploración (`notebooks/exploracion/`)

Notebooks experimentales desarrollados durante el proceso de exploración:
- **baseline.ipynb**, **baseline_hmade.ipynb**, **baseline_intento_2.ipynb** - Modelos baseline iniciales
- **baseline_mobilenet.ipynb**, **baseline_mobilenet_mixed_dataset.ipynb** - Experimentos con MobileNet
- **baseline_mobilenet_augmented_dataset.ipynb** - MobileNet con data augmentation
- **mobilenet_augmented_dataset.ipynb** - Refinamiento de augmentation
- **comet_process_data.ipynb** - Procesamiento de datos con logging de experimentos

### Otros Archivos

- **Propuesta de Planificación.md** - Documento de planificación inicial del proyecto