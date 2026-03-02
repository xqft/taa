# Proyecto Kaggle - Higgs Boson Challenge

**Tipo:** Clasificación binaria en física de partículas
**Dataset:** CERN - Eventos del Large Hadron Collider
**Objetivo:** Identificar señales del bosón de Higgs entre ruido de fondo
**Competencia:** [Higgs Boson Machine Learning Challenge](https://www.kaggle.com/c/higgs-boson)

## Descripción

Este proyecto forma parte del Taller de Aprendizaje Automático y se enfoca en el análisis del conjunto de datos proporcionado por el CERN para la competencia del Bosón de Higgs en Kaggle. El desafío consiste en distinguir señales del bosón de Higgs de eventos de ruido de fondo utilizando características de colisiones de partículas del Large Hadron Collider.

Los datos contienen características físicas derivadas de eventos de colisión, y el objetivo es maximizar la métrica AMS (Approximate Median Significance), que mide la significancia estadística del descubrimiento.

## Metodología

El proyecto explora múltiples enfoques de aprendizaje automático:

### Modelos Implementados
- **XGBoost**: Modelo de gradient boosting con optimización de hiperparámetros
- **Redes Neuronales**: Modelos de deep learning con Keras/TensorFlow y Hyperband tuning
- **Ensemble Methods**: Exploración de múltiples algoritmos de boosting y bagging

### Técnicas Aplicadas
- Análisis exploratorio de datos (EDA)
- Ingeniería de características
- Optimización de threshold de clasificación
- Validación cruzada
- Optimización de hiperparámetros (Keras Tuner, Hyperband)
- Análisis de curvas ROC y precision-recall

### Notebooks
- `notebooks/AnalisisDeDatos.ipynb` - Exploración y análisis del dataset
- `notebooks/XGBoost.ipynb` - Implementación y optimización de XGBoost
- `notebooks/red_neuronal.ipynb` - Redes neuronales con Keras
- `notebooks/mejor_modelo.ipynb` - Consolidación del mejor modelo

## Resultados

El proyecto logró optimizar la métrica AMS a través de múltiples iteraciones, explorando diferentes arquitecturas y técnicas de preprocesamiento. Los resultados detallados se encuentran en el informe oficial.

**Ver informe completo:** [`docs/higgs-boson-informe.pdf`](../docs/higgs-boson-informe.pdf)

## Tecnologías

- **Python 3.x**
- **scikit-learn** - Modelos clásicos de ML
- **XGBoost** - Gradient boosting
- **TensorFlow/Keras** - Redes neuronales
- **pandas, NumPy** - Manipulación de datos
- **matplotlib, seaborn** - Visualización
- **Keras Tuner** - Optimización de hiperparámetros
- **imbalanced-learn** - Técnicas de oversampling

## Configuración Inicial

### Requisitos
```bash
# Instalar dependencias desde el root del repositorio
pip install -r requirements.txt
```

### Descarga de Datos

Antes de comenzar, es necesario estar registrado en [Kaggle](https://www.kaggle.com) y haber aceptado las condiciones de la competencia.

1. Configurar credenciales (descargar `kaggle.json` desde tu perfil de Kaggle):
   ```bash
   mkdir -p ~/.kaggle/
   mv kaggle.json ~/.kaggle/
   chmod 600 ~/.kaggle/kaggle.json
   ```

2. Descargar el dataset:
   ```bash
   kaggle competitions download -c higgs-boson
   unzip higgs-boson.zip -d data/
   ```

## Estructura del Proyecto

```
06-kaggle-higgs-boson/
├── README.md
├── notebooks/
│   ├── AnalisisDeDatos.ipynb      # Exploración y análisis del dataset
│   ├── XGBoost.ipynb              # Implementación de XGBoost
│   ├── red_neuronal.ipynb         # Redes neuronales con Keras
│   └── mejor_modelo.ipynb         # Mejor modelo final
├── utils/                         # Utilidades auxiliares
├── constantes.py                  # Constantes del proyecto
├── preprocess.py                  # Funciones de preprocesamiento
├── plot_functions.py              # Utilidades de visualización
└── HiggsBosonCompetition_AMSMetric_rev1.py  # Métrica AMS
```

## Ejecución

```bash
# Ejecutar análisis exploratorio
jupyter notebook notebooks/AnalisisDeDatos.ipynb

# Entrenar modelo XGBoost
jupyter notebook notebooks/XGBoost.ipynb

# Explorar mejor modelo
jupyter notebook notebooks/mejor_modelo.ipynb
```

## Métricas

El proyecto utiliza la métrica **AMS (Approximate Median Significance)**, específica de la competencia, que mide la significancia estadística del descubrimiento del bosón de Higgs. Esta métrica balancea la tasa de verdaderos positivos con la tasa de falsos positivos, fundamental en física de partículas.

## Autores

Proyecto desarrollado como parte del Taller de Aprendizaje Automático - Facultad de Ingeniería, Universidad de la República (Uruguay)
