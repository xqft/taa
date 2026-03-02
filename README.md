# Taller de Aprendizaje Automático (TAA)

**Facultad de Ingeniería, Universidad de la República**
**Curso 2024**

Este repositorio consolida los proyectos individuales desarrollados durante el curso de Taller de Aprendizaje Automático, basado en el libro "Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow" de Aurélien Géron.

## Estructura del Repositorio

### Talleres Semanales

Cinco talleres prácticos desarrollados durante el semestre, cada uno explorando diferentes técnicas de machine learning:

- **[01-clasificador-titanic](./01-clasificador-titanic/)** - Clasificador binario de supervivientes del Titanic
- **[02-sentiment-imdb](./02-sentiment-imdb/)** - Análisis de sentimiento en reseñas de películas
- **[03-regresion-bicicletas](./03-regresion-bicicletas/)** - Predicción de demanda de bicicletas (Random Forest + XGBoost)
- **[04-deteccion-anomalias](./04-deteccion-anomalias/)** - Detección de anomalías en datasets
- **[05-bicicletas-rnn](./05-bicicletas-rnn/)** - Predicción temporal con redes neuronales recurrentes

### Proyectos Kaggle

Dos proyectos competitivos desarrollados en el curso, con código completo y documentación:

6. **[06-kaggle-higgs-boson](./06-kaggle-higgs-boson/)** - Clasificación de eventos de física de partículas ([Informe](./docs/higgs-boson-informe.pdf))
7. **[07-kaggle-freesound](./07-kaggle-freesound/)** - Clasificación de audio con deep learning ([Informe](./docs/freesound-informe.pdf))

### Documentación

La carpeta `docs/` contiene los informes en PDF de todos los proyectos:

- `higgs-boson-informe.pdf` - Proyecto 1: Higgs Boson Challenge
- `freesound-informe.pdf` - Proyecto 2: Freesound Audio Tagging
- `entregable-1.pdf` - Entregable del primer proyecto
- `entregable-2.pdf` - Entregable del segundo proyecto

## Resultados Destacados

- **Sentiment Analysis (IMDB):** 87% accuracy con TF-IDF, bigramas y eliminación de stopwords
- **Bike Demand (XGBoost):** RMSLE ~0.37 en validación cruzada (top 5% en Kaggle ~0.35)
- **Titanic Classifier:** Identificación de clase social y género como variables críticas de supervivencia

## Configuración del Entorno

```bash
# Crear entorno conda con todas las dependencias
conda env create -f environment.yml
conda activate taa

# Iniciar Jupyter para explorar notebooks
jupyter notebook
```

## Tecnologías Utilizadas

- **Frameworks:** scikit-learn, Keras, TensorFlow, PyTorch
- **Procesamiento:** pandas, numpy, matplotlib, seaborn
- **Modelos:** Random Forest, XGBoost, RNN/LSTM, transformers
- **Métricas:** TF-IDF, SHAP values, validación cruzada
