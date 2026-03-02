# Predicción de Demanda de Bicicletas

**Taller 3 - Regresión con Random Forest y Gradient Boosting**

## Descripción

Modelo de regresión para predecir la demanda de bicicletas de alquiler basándose en condiciones climáticas, temporales y estacionales. Proyecto enfocado en técnicas avanzadas de ensemble learning y feature engineering.

## Dataset

**Fuente:** Kaggle Bike Sharing Demand
- **Período:** 2 años de datos históricos
- **Granularidad:** Horaria
- **Features:** Fecha/hora, temperatura, humedad, velocidad del viento, estacionalidad, días festivos

Descargar el dataset de [Kaggle Bike Sharing Demand](https://www.kaggle.com/c/bike-sharing-demand/data) y colocar los archivos en `data/`.

## Resultados

**RMSLE en validación cruzada: ~0.37**

> **Contexto:** El top 5% del leaderboard de Kaggle tiene RMSLE ~0.35

### Modelos Probados

1. **Random Forest** (baseline): RMSLE ~0.45
2. **XGBoost** (final): RMSLE ~0.37
3. **Gradient Boosting Regressor**: RMSLE ~0.39

## Feature Engineering

**Variables derivadas creadas:**
- Hora del día (picos de commute vs horas valle)
- Día de la semana (weekday vs weekend)
- Estación del año
- Interacciones temperatura × humedad
- Flags para condiciones climáticas extremas

**Análisis SHAP:**
- La hora del día es el factor más influyente
- Temperatura tiene relación no lineal con demanda
- Días laborables vs fines de semana presentan patrones distintos

## Técnicas Utilizadas

- **XGBoost:** Gradient Boosting optimizado para velocidad y rendimiento
- **Random Forest:** Ensemble de árboles de decisión
- **SHAP Values:** Interpretabilidad del modelo (ver `shap.png`)
- **Validación cruzada:** K-fold para evaluar generalización
- **Hyperparameter tuning:** Grid search para max_depth, learning_rate, n_estimators

## Visualizaciones

Las siguientes visualizaciones se generan al ejecutar el notebook:

- Relación entre temperatura y demanda
- Impacto de max_depth en rendimiento
- SHAP summary plot de importancia de features

## Archivos

- `taller3_demanda_de_bicicletas.ipynb` - Notebook principal
- `best_model_t3.py` - Script con mejor configuración del modelo
- `comet_log.py` - Logging de experimentos (integración con Comet.ml)
- `data/` - Datasets de Kaggle

## Ejecutar el Proyecto

```bash
conda activate taa
jupyter notebook taller3_demanda_de_bicicletas.ipynb
```

## Aprendizajes Clave

- **Ensemble methods superan modelos individuales** - XGBoost > Random Forest > Regresión lineal
- **Feature engineering es crítico** - Variables derivadas mejoran RMSLE en ~20%
- **RMSLE penaliza errores en valores bajos** - Importante balancear predicciones en rangos
- **Interpretabilidad importa** - SHAP permite explicar decisiones del modelo
- **Overfitting es un riesgo real** - Cross-validation detecta sobreajuste

## Próximos Pasos

- Experimentar con redes neuronales (ver taller 5 para enfoque temporal)
- Incorporar datos externos (clima histórico, eventos locales)
- Ensembles de modelos (stacking XGBoost + LSTM)
- Feature selection automático (recursive feature elimination)
