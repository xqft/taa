# Predicción con Redes Neuronales Recurrentes

## Descripción

Predicción de demanda de alquiler de bicicletas utilizando **Redes Neuronales Recurrentes (RNN)** para capturar patrones temporales y dependencias secuenciales en los datos. Este proyecto aplica deep learning a series temporales.

## Dataset

Mismo dataset del proyecto de regresión de bicicletas ([Bike Sharing Demand de Kaggle](https://www.kaggle.com/c/bike-sharing-demand)), pero abordado con arquitecturas de redes neuronales diseñadas para datos secuenciales.

## Metodología

### Arquitectura de Red
- **RNN (Recurrent Neural Network)**: Captura dependencias temporales
- Posiblemente **LSTM/GRU**: Variantes avanzadas para secuencias largas
- Capas densas para la predicción final

### Preprocesamiento para RNNs
- Normalización de características
- Creación de ventanas temporales (sequences)
- Preparación de datos en formato 3D para RNNs

### Entrenamiento
- Framework: TensorFlow/Keras
- Optimización con callbacks (early stopping, learning rate scheduling)
- Validación temporal (respetando el orden cronológico)

## Resultados

- Comparación con modelos tradicionales (Random Forest, XGBoost)
- Análisis de la capacidad de las RNNs para capturar patrones temporales
- Evaluación con RMSLE
- Visualización de predicciones vs valores reales

## Tecnologías

- **TensorFlow/Keras**: Construcción y entrenamiento de RNNs
- **pandas**: Manipulación de series temporales
- **NumPy**: Operaciones numéricas y creación de secuencias
- **matplotlib**: Visualización de predicciones

## Archivos

- `taller5_demanda_de_bicicletas_con_NNs.ipynb`: Notebook principal
- `train_model.py`: Script de entrenamiento
- `data/train.csv`: Datos de entrenamiento
- `data/test.csv`: Datos de prueba
- `data/submission.csv`: Predicciones para Kaggle
- Posibles archivos de modelos guardados (`.h5`, `.pkl`)

## Comparación con Proyecto 03

| Aspecto | Proyecto 03 (ML Tradicional) | Proyecto 05 (RNNs) |
|---------|----------------------------|-------------------|
| Modelo | Random Forest, XGBoost | RNN/LSTM/GRU |
| Enfoque | Features engineered | Aprendizaje de secuencias |
| Ventaja | Interpretabilidad, rapidez | Captura patrones temporales |
| Complejidad | Media | Alta |

## Aprendizajes Clave

- Arquitecturas de RNNs para series temporales
- Preparación de datos secuenciales
- Trade-offs entre modelos tradicionales y deep learning
- Importancia del diseño de ventanas temporales
- Técnicas de regularización en redes neuronales

## Ejecución

```bash
# Desde el directorio del proyecto
jupyter notebook taller5_demanda_de_bicicletas_con_NNs.ipynb

# O ejecutar el script de entrenamiento
python train_model.py
```

---

[← Volver al repositorio principal](../)
