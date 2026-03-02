# Predicción de Demanda con Redes Neuronales Recurrentes

**Taller 5 - Deep Learning y Series Temporales**

## Descripción

Predicción de demanda de bicicletas utilizando Redes Neuronales Recurrentes (RNN) con arquitectura LSTM. Este proyecto explora el enfoque de deep learning para series temporales, contrastando con el enfoque de ensemble learning del Taller 3.

## Motivación

A diferencia del Taller 3 (que trató cada observación independientemente con XGBoost), este proyecto modela la **dependencia temporal**:
- Patrones de demanda se repiten día a día
- Tendencias semanales y estacionales
- Autocorrelación en series temporales

## Dataset

**Mismo dataset que Taller 3:** Kaggle Bike Sharing Demand
- 2 años de datos horarios
- Features climáticas y temporales

Descargar el dataset de [Kaggle Bike Sharing Demand](https://www.kaggle.com/c/bike-sharing-demand/data) y colocar los archivos en `data/`.

## Arquitectura del Modelo

### LSTM (Long Short-Term Memory)

```
Input Sequence (lookback window)
    ↓
LSTM Layer 1 (128 units)
    ↓
Dropout (0.2)
    ↓
LSTM Layer 2 (64 units)
    ↓
Dropout (0.2)
    ↓
Dense Output Layer
```

**Hiperparámetros clave:**
- **Lookback window:** 24-48 horas (cuánta historia usar)
- **Batch size:** 32-64
- **Epochs:** 50-100 con early stopping
- **Optimizer:** Adam
- **Loss:** MSE (Mean Squared Error)

## Preprocesamiento para RNN

1. **Normalización:** MinMaxScaler para estabilizar entrenamiento
2. **Secuenciación:** Crear ventanas deslizantes de observaciones
3. **Train/validation split:** Mantener orden temporal (no shuffle)
4. **Feature engineering:** Variables cíclicas (hora, día) con sin/cos encoding

## Resultados

**Comparación con Taller 3:**
- XGBoost (Taller 3): RMSLE ~0.37
- LSTM (Taller 5): Resultados comparables, mayor capacidad de capturar tendencias

**Ventajas de LSTM:**
- Captura patrones temporales automáticamente
- No requiere feature engineering manual
- Generaliza bien a nuevas secuencias

**Desventajas de LSTM:**
- Mayor tiempo de entrenamiento
- Requiere más datos
- Menos interpretable

## Técnicas de Deep Learning

- **LSTM Cells:** Memoria a largo plazo con gates (forget, input, output)
- **Dropout:** Regularización para prevenir overfitting
- **Early Stopping:** Detener entrenamiento al detectar overfitting
- **Batch Normalization:** Estabilizar entrenamiento
- **Learning Rate Scheduling:** Ajustar tasa de aprendizaje dinámicamente

## Visualizaciones

- Predicciones vs valores reales
- Curvas de aprendizaje (loss en train/validation)
- Autocorrelación de residuos

## Archivos

- `taller5_bike_demand_rnn.ipynb` - Notebook con implementación LSTM
- `data/` - Dataset de Kaggle
- Modelos guardados (si aplicable)

## Ejecutar el Proyecto

```bash
conda activate taa
jupyter notebook taller5_bike_demand_rnn.ipynb
```

**Nota:** Entrenamiento puede tomar tiempo. Considerar usar GPU si está disponible.

## Aprendizajes Clave

- **LSTMs son poderosas para secuencias** - Superan modelos tradicionales en datos temporales
- **Preprocesamiento diferente** - Secuenciación y normalización críticas
- **Tradeoff interpretabilidad-rendimiento** - LSTM más potente pero menos explicable
- **Overfitting es un riesgo mayor** - Dropout y early stopping esenciales
- **GPU acelera significativamente** - Diferencia de horas vs minutos

## Comparación: XGBoost vs LSTM

| Aspecto | XGBoost (Taller 3) | LSTM (Taller 5) |
|---------|-------------------|-----------------|
| **Feature Engineering** | Manual, intensivo | Automático |
| **Interpretabilidad** | Alta (SHAP) | Baja |
| **Tiempo de entrenamiento** | Minutos | Horas |
| **Captura temporal** | Limitada (features manuales) | Excelente |
| **Generalización** | Buena | Muy buena |
| **Datos requeridos** | Moderados | Muchos |

## Extensiones Posibles

- **GRU (Gated Recurrent Units)** - Alternativa más simple a LSTM
- **Bidirectional LSTM** - Contexto futuro y pasado
- **Attention Mechanisms** - Focalizar en momentos importantes
- **Transformers** - Estado del arte en secuencias
- **Multivariate forecasting** - Predecir múltiples variables simultáneamente
- **Ensemble** - Combinar XGBoost + LSTM para mejor rendimiento

## Recursos

- [Understanding LSTM Networks](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- [Keras Time Series Tutorial](https://www.tensorflow.org/tutorials/structured_data/time_series)
- Géron - Capítulo 15: Processing Sequences Using RNNs and CNNs
