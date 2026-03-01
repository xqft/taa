# Clasificador de Supervivencia del Titanic

## Descripción

Desarrollo de un clasificador binario para predecir la supervivencia de pasajeros del Titanic basándose en características personales. Este proyecto introduce conceptos fundamentales de aprendizaje automático incluyendo exploración de datos, ingeniería de características y construcción de pipelines.

## Dataset

Conjunto de datos del [desafío Titanic de Kaggle](https://www.kaggle.com/c/titanic) con información de 891 pasajeros incluyendo:
- Características demográficas (edad, sexo)
- Clase socioeconómica (clase del boleto)
- Información familiar (hermanos, padres, hijos a bordo)
- Tarifa y puerto de embarque

## Metodología

### Exploración de Datos
- Análisis de datos faltantes (Age: 177 valores, Cabin: 687 valores)
- Estudio de correlación entre características
- Visualización de distribuciones por clase de supervivencia

### Preprocesamiento
Implementé dos pipelines usando `scikit-learn`:

1. **Pipeline completo**: Todas las características excepto Cabin, Name y Ticket
   - Imputación de valores faltantes (media para Age, moda para Embarked)
   - One-hot encoding para variables categóricas
   - Accuracy: **78.23%**

2. **Pipeline de sexo únicamente**: Solo la característica Sex
   - Accuracy: **78.67%**

### Modelo
- **Regresión Logística** con parámetros por defecto
- Validación cruzada 5-folds
- Búsqueda de hiperparámetros con Grid Search para optimizar el parámetro C

## Resultados Clave

### Factores de Supervivencia
1. **Sexo**: Factor más determinante (pipeline solo con sexo alcanza 78.67% accuracy)
2. **Clase del pasajero**: Fuerte correlación (correlación absoluta: 0.338)
   - Primera clase: 62.96% de supervivencia
   - Segunda clase: 47.28% de supervivencia
   - Tercera clase: 24.24% de supervivencia
3. **Tarifa**: Correlación moderada (0.257)

### Observaciones
- Las mujeres fueron priorizadas en los botes salvavidas
- La clase socioeconómica tuvo un impacto significativo en las posibilidades de supervivencia
- El dataset presenta desbalanceo moderado (38.38% de supervivientes)

## Tecnologías

- **pandas**: Exploración y manipulación de datos
- **scikit-learn**: Pipelines, preprocesamiento y modelado
- **matplotlib**: Visualización de datos
- **Kaggle API**: Descarga de datasets y submissions

## Archivos

- `taller1_titanic.ipynb`: Notebook principal con análisis completo
- `data/train.csv`: Conjunto de entrenamiento
- `data/test.csv`: Conjunto de prueba
- `data/gender_submission.csv`: Ejemplo de submission

## Ejecución

```bash
# Desde el directorio del proyecto
jupyter notebook taller1_titanic.ipynb
```

---

[← Volver al repositorio principal](../)
