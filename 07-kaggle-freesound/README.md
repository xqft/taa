# Laboratorio 2 - Freesound Audio Tagging Challenge

**Tipo:** Clasificación multiclase de audio
**Dataset:** Freesound - Miles de clips de audio etiquetados
**Objetivo:** Clasificar clips de audio en 80 categorías diferentes
**Competencia:** [Freesound Audio Tagging Challenge](https://www.kaggle.com/c/freesound-audio-tagging)

## Descripción

Este laboratorio forma parte del Taller de Aprendizaje Automático y aborda el desafío de clasificación de audio multiclase del dataset Freesound. El objetivo es desarrollar modelos capaces de etiquetar automáticamente clips de audio cortos en una de 80 categorías posibles.

La competencia utiliza la métrica **label-weighted label-ranking average precision** (lwlrap), que mide la capacidad del modelo para rankear correctamente las etiquetas verdaderas por encima de las falsas, considerando el desbalance de clases.

## Metodología

El proyecto explora arquitecturas de deep learning para procesamiento de audio:

### Enfoque Principal
- **Transfer Learning con MobileNet**: Arquitectura eficiente para clasificación de imágenes aplicada a espectrogramas de audio
- **Espectrogramas**: Conversión de señales de audio a representaciones visuales (mel-spectrograms)
- **Data Augmentation**: Técnicas de aumentación de datos específicas para audio

### Técnicas Aplicadas
- Análisis exploratorio de datos (EDA)
- Conversión de audio a espectrogramas (librosa)
- Preprocesamiento de audio:
  - Resizing de clips
  - Normalización
  - Data augmentation (velocidad, ruido, pitch, volumen)
- Transfer learning con arquitecturas preentrenadas:
  - **MobileNet**: Arquitectura ligera y eficiente
  - **VGG16**: Modelo clásico para clasificación
  - **ResNet**: Redes residuales profundas
  - **EfficientNet**: Escalado balanceado de redes
- Optimización de hiperparámetros (Optuna, Keras Tuner)
- Técnicas de regularización (Dropout, Batch Normalization, EarlyStopping)
- Experimentación con Comet ML

### Notebooks
- `EDA.ipynb` - Análisis exploratorio del dataset de audio
- `notebooks/procesamiento_de_audios.ipynb` - Pipeline de preprocesamiento de audio
- `notebooks/best_mobilenet.ipynb` - Mejor modelo MobileNet optimizado

## Resultados

El proyecto logró desarrollar un modelo robusto de clasificación de audio utilizando transfer learning y data augmentation. Los experimentos exploraron diferentes arquitecturas y técnicas de preprocesamiento para maximizar la métrica lwlrap.

**Ver informe completo:** [`docs/freesound-informe.pdf`](../docs/freesound-informe.pdf)

## Tecnologías

- **Python 3.x**
- **TensorFlow/Keras** - Deep learning framework
- **librosa** - Procesamiento y análisis de audio
- **pandas, NumPy** - Manipulación de datos
- **matplotlib, seaborn** - Visualización
- **Comet ML** - Tracking de experimentos
- **Optuna / Keras Tuner** - Optimización de hiperparámetros
- **AWS** - Entrenamiento en la nube (GPU)

## Configuración Inicial

### Requisitos
```bash
# Instalar dependencias desde el root del repositorio
pip install -r requirements.txt
```

### Descarga de Datos

1. Registrarse en [Kaggle](https://www.kaggle.com) y aceptar las condiciones de la competencia

2. Configurar Kaggle API:
   ```bash
   mkdir -p ~/.kaggle/
   mv kaggle.json ~/.kaggle/
   chmod 600 ~/.kaggle/kaggle.json
   ```

3. Descargar el dataset:
   ```bash
   kaggle competitions download -c freesound-audio-tagging
   unzip freesound-audio-tagging.zip -d data/
   ```

## Estructura del Proyecto

```
07-kaggle-freesound/
├── README.md
├── constantes.py                    # Constantes del proyecto
├── EDA.ipynb                        # Análisis exploratorio
├── notebooks/
│   ├── procesamiento_de_audios.ipynb  # Pipeline de preprocesamiento
│   └── best_mobilenet.ipynb           # Mejor modelo
└── utils/                           # Utilidades auxiliares
```

## Ejecución

```bash
# Análisis exploratorio
jupyter notebook EDA.ipynb

# Procesamiento de audio
jupyter notebook notebooks/procesamiento_de_audios.ipynb

# Mejor modelo
jupyter notebook notebooks/best_mobilenet.ipynb
```

## Métricas

La competencia utiliza **lwlrap (label-weighted label-ranking average precision)**, que:
- Evalúa el ranking de las predicciones
- Considera el desbalance de clases
- Penaliza predicciones incorrectas rankeadas más alto que las correctas
- Es especialmente apropiada para clasificación multiclase con muchas categorías

## Pipeline de Audio

1. **Carga de audio**: Leer archivos .wav con librosa
2. **Conversión a espectrograma**: Transformar señal temporal a representación frecuencial (mel-spectrogram)
3. **Augmentation** (opcional):
   - Time stretching (cambio de velocidad)
   - Pitch shifting (cambio de tono)
   - Adición de ruido
   - Cambio de volumen
4. **Normalización**: Estandarizar valores de entrada
5. **Clasificación**: Pasar espectrograma por CNN (MobileNet)

## Autores

Proyecto desarrollado como parte del Taller de Aprendizaje Automático - Facultad de Ingeniería, Universidad de la República (Uruguay)
