# Análisis de Sentimiento - Reseñas IMDB

## Descripción

Clasificador binario de sentimiento para reseñas de películas de IMDB. Este proyecto aplica técnicas de procesamiento de lenguaje natural (NLP) para determinar si una reseña es positiva o negativa.

## Dataset

50,000 reseñas de películas del [dataset de IMDB](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews):
- 25,000 reseñas positivas
- 25,000 reseñas negativas
- Dataset balanceado ideal para clasificación binaria

## Metodología

### Preprocesamiento de Texto
- Eliminación de stop words (palabras comunes sin significado semántico)
- Generación de bigrams para capturar contexto local
- Vectorización con **TF-IDF** (Term Frequency-Inverse Document Frequency)

### Técnicas Aplicadas
1. **Vectorización TF-IDF**: Ponderación de términos según su importancia
2. **N-grams**: Uso de bigrams para capturar secuencias de palabras
3. **Stop words removal**: Filtrado de palabras sin valor semántico

### Modelo
- Clasificador de aprendizaje automático (detalles en el notebook)
- Métricas de evaluación: accuracy, precision, recall, F1-score

## Resultados

- **Accuracy en test set: 87%**
- Métricas adicionales evaluadas: precision, recall y F1-score
- El uso de bigrams y TF-IDF mejora significativamente el rendimiento

## Tecnologías

- **pandas**: Manipulación de datos textuales
- **scikit-learn**: Vectorización TF-IDF y modelado
- **NLTK/spaCy**: Procesamiento de lenguaje natural
- **matplotlib**: Visualización de resultados

## Archivos

- `taller2_criticas_cine.ipynb`: Notebook con análisis completo
- `data/IMDB Dataset.csv`: Dataset de 50k reseñas

## Aprendizajes Clave

- La importancia del preprocesamiento en NLP
- Cómo los bigrams capturan mejor el contexto que unigrams solos
- El impacto de TF-IDF en la mejora del rendimiento
- Balance entre complejidad del modelo y generalización

## Ejecución

```bash
# Desde el directorio del proyecto
jupyter notebook taller2_criticas_cine.ipynb
```

---

[← Volver al repositorio principal](../)
