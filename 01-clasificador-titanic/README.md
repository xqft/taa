# Clasificador de Supervivientes del Titanic

**Taller 1 - Clasificación Binaria**

## Descripción

Clasificador binario que predice la supervivencia de pasajeros del Titanic basándose en características demográficas y socioeconómicas. Este proyecto introduce conceptos fundamentales de machine learning supervisado y análisis exploratorio de datos.

## Dataset

**Fuente:** Kaggle Titanic Competition
- **Registros:** ~900 pasajeros
- **Features:** Clase, sexo, edad, número de familiares, tarifa, puerto de embarque
- **Target:** Supervivencia (0 = No sobrevivió, 1 = Sobrevivió)

Descargar el dataset de [Kaggle Titanic](https://www.kaggle.com/c/titanic/data) y colocar los archivos en `data/`:
- `train.csv` - Conjunto de entrenamiento
- `test.csv` - Conjunto de prueba
- `gender_submission.csv` - Ejemplo de formato de submission

## Hallazgos Principales

**Variables más significativas:**
1. **Clase del pasajero** - Los pasajeros de primera clase tuvieron mayor tasa de supervivencia
2. **Sexo** - Las mujeres fueron priorizadas en los botes salvavidas ("mujeres y niños primero")
3. **Edad** - Los niños tuvieron mayor probabilidad de supervivencia

**Insights:**
- La política de evacuación ("mujeres y niños primero") se refleja claramente en los datos
- El estatus socioeconómico (clase de ticket) tuvo impacto significativo en las probabilidades de supervivencia
- La combinación de estas variables permite una predicción razonablemente precisa

## Técnicas Utilizadas

- Análisis exploratorio de datos (EDA)
- Manejo de valores faltantes
- Encoding de variables categóricas
- Validación cruzada
- Métricas de clasificación: accuracy, precision, recall

## Archivos

- `taller1_titanic.ipynb` - Notebook principal con análisis y modelo
- `data/` - Datasets del desafío Titanic

## Ejecutar el Proyecto

```bash
# Activar entorno
conda activate taa

# Iniciar Jupyter
jupyter notebook taller1_titanic.ipynb
```

## Contexto del Curso

Este taller introduce:
- Flujo completo de un proyecto de ML supervisado
- Importancia del análisis exploratorio
- Balance entre interpretabilidad y rendimiento
- Evaluación de modelos de clasificación
