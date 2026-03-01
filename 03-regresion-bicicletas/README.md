# Regresión - Demanda de Alquiler de Bicicletas

## Descripción

Proyecto de regresión para predecir la demanda de alquiler de bicicletas basándose en condiciones meteorológicas y temporales. Participación en el [desafío Bike Sharing Demand de Kaggle](https://www.kaggle.com/c/bike-sharing-demand).

## Dataset

Datos históricos de Capital Bikeshare (Washington D.C.) con características:
- **Temporales**: fecha, hora, día de la semana, festivos
- **Meteorológicas**: temperatura, humedad, velocidad del viento
- **Estacionales**: estación del año
- **Target**: conteo de bicicletas alquiladas

## Metodología

### Ingeniería de Características
- Extracción de componentes temporales (hora, día, mes)
- Análisis de patrones estacionales
- Transformaciones de variables meteorológicas

### Modelos Utilizados
1. **Random Forest**: Ensemble de árboles de decisión
2. **XGBoost**: Gradient boosting optimizado
   - Mejor rendimiento en validación cruzada
   - Hiperparámetros optimizados con Grid Search

### Evaluación
- Métrica: **RMSLE** (Root Mean Squared Logarithmic Error)
- RMSLE en validación cruzada: **~0.37**
- Top 5 del leaderboard de Kaggle: ~0.35 RMSLE

## Resultados

### Características Más Importantes
Análisis con SHAP (SHapley Additive exPlanations):
- Hora del día
- Temperatura
- Día de la semana
- Estación del año
- Condiciones climáticas

### Rendimiento del Modelo
- **RMSLE Cross-validation: 0.37**
- Competitivo con soluciones top del leaderboard
- Balance entre complejidad y generalización

## Visualizaciones

El proyecto incluye:
- Análisis de importancia de características
- Gráficos SHAP para interpretabilidad
- Curvas de validación para `max_depth` y otros hiperparámetros
- Visualización de patrones temporales en la demanda

## Tecnologías

- **pandas**: Manipulación de datos temporales
- **scikit-learn**: Preprocesamiento y Random Forest
- **XGBoost**: Gradient boosting
- **SHAP**: Interpretabilidad del modelo
- **matplotlib/seaborn**: Visualización

## Archivos

- `taller3_demanda_de_bicicletas.ipynb`: Notebook principal
- `best_model_t3.py`: Script del mejor modelo
- `comet_log.py`: Logging de experimentos
- `data/train.csv`: Datos de entrenamiento
- `data/test.csv`: Datos de prueba
- `data/submission.csv`: Predicciones para Kaggle
- `*.png`: Visualizaciones generadas

## Aprendizajes Clave

- Importancia de la ingeniería de características en problemas temporales
- XGBoost como herramienta poderosa para regresión
- Uso de SHAP para interpretar modelos complejos
- Optimización de hiperparámetros con Grid Search

## Ejecución

```bash
# Desde el directorio del proyecto
jupyter notebook taller3_demanda_de_bicicletas.ipynb
```

---

[← Volver al repositorio principal](../)
