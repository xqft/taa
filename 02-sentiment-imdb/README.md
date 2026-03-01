# Análisis de Sentimiento - Reseñas de IMDB

**Taller 2 - Procesamiento de Lenguaje Natural (NLP)**

## Descripción

Clasificador de sentimiento binario para reseñas de películas de IMDB. El modelo determina si una reseña es positiva o negativa utilizando técnicas de procesamiento de lenguaje natural y machine learning.

## Dataset

**Fuente:** IMDB Movie Reviews Dataset
- **Registros:** 50,000 reseñas
- **Clases:** Positivo/Negativo (balanceado 50/50)
- **Características:** Texto libre en inglés

Archivos en `data/`:
- `IMDB Dataset.csv` - Dataset completo
- `imdb-dataset-of-50k-movie-reviews.zip` - Archivo comprimido

## Resultados

**Accuracy final: 87% en conjunto de prueba**

### Pipeline de Procesamiento

1. **Limpieza de texto**
   - Eliminación de stopwords
   - Lowercasing
   - Tokenización

2. **Vectorización**
   - TF-IDF (Term Frequency - Inverse Document Frequency)
   - Bigramas para capturar contexto
   - Peso de términos por relevancia

3. **Clasificación**
   - Modelos probados: Naive Bayes, SVM, Logistic Regression
   - Mejor resultado: SVM con TF-IDF + bigramas

## Técnicas de NLP Utilizadas

- **TF-IDF:** Ponderación de términos por frecuencia e importancia
- **N-gramas:** Bigramas para capturar frases significativas ("not good", "very bad")
- **Stopwords:** Eliminación de palabras comunes sin valor semántico
- **Bag of Words:** Representación vectorial del texto

## Desafíos

- **Negaciones:** "not good" vs "good" requiere contexto (resuelto con bigramas)
- **Sarcasmo:** Difícil de detectar con técnicas básicas
- **Palabras ambiguas:** Algunos términos cambian significado según contexto

## Archivos

- `taller2_criticas_cine.ipynb` - Notebook con análisis y modelo
- `data/` - Dataset de reseñas IMDB
- `.ipynb_checkpoints/` - Checkpoints de Jupyter

## Ejecutar el Proyecto

```bash
conda activate taa
jupyter notebook taller2_criticas_cine.ipynb
```

## Aprendizajes Clave

- La importancia de la vectorización en NLP
- TF-IDF es más efectivo que simple Bag of Words
- Los bigramas mejoran significativamente la captura de contexto
- Las stopwords pueden eliminarse sin pérdida significativa de información
- El balanceo del dataset es crucial para métricas confiables

## Posibles Mejoras

- Incorporar word embeddings (Word2Vec, GloVe)
- Experimentar con modelos de deep learning (LSTM, transformers)
- Análisis de errores en predicciones incorrectas
- Aumentar el dataset con técnicas de data augmentation
