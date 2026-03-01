# Taller de Aprendizaje Automático

**Facultad de Ingeniería, UdelaR - 2024**

Colección de proyectos desarrollados en el curso Taller de Aprendizaje Automático, basado en el libro "Hands-On Machine Learning" de Aurélien Géron.

## 🎯 Habilidades Demostradas

- Clasificación binaria y multiclase
- Regresión y predicción de series temporales
- Procesamiento de lenguaje natural (NLP)
- Redes neuronales recurrentes (RNN)
- Detección de anomalías
- Deep learning con TensorFlow/Keras
- Ingeniería de características y preprocesamiento de datos
- Optimización de hiperparámetros con Grid Search
- Validación cruzada y evaluación de modelos

## 🛠️ Tecnologías

Python, scikit-learn, TensorFlow, Keras, XGBoost, pandas, NumPy, matplotlib, seaborn

## 📂 Proyectos

### Talleres Semanales

1. **[Clasificador Titanic](./01-clasificador-titanic/)** - Predicción de supervivencia usando regresión logística. Demostré que el sexo y la clase del pasajero fueron los factores más significativos.

2. **[Análisis de Sentimiento IMDB](./02-sentiment-imdb/)** - Clasificación binaria de reseñas de películas. Alcancé 87% de accuracy usando stop words, bigrams y tf-idf.

3. **[Regresión Alquiler de Bicicletas](./03-regresion-bicicletas/)** - Predicción de demanda usando Random Forest con XGBoost. RMSLE de ~0.37 en validación cruzada.

4. **[Detección de Anomalías](./04-deteccion-anomalias/)** - Identificación de patrones anómalos en datos utilizando técnicas de aprendizaje no supervisado.

5. **[Predicción con RNN](./05-bicicletas-rnn/)** - Series temporales con redes neuronales recurrentes para predicción de demanda de bicicletas.

### Proyectos Kaggle

Los siguientes proyectos fueron desarrollados como laboratorios del curso, participando en competencias de Kaggle:

6. **Higgs Boson Challenge** - Clasificación de eventos de física de partículas ([Ver informe](./docs/higgs-boson-informe.pdf))

7. **Freesound Audio Tagging** - Clasificación de audio usando técnicas de deep learning ([Ver informe](./docs/freesound-informe.pdf))

## 📄 Informes

- [Proyecto 1 - Higgs Boson](./docs/higgs-boson-informe.pdf)
- [Laboratorio 2 - Freesound](./docs/freesound-informe.pdf)
- [Entregable 1](./docs/entregable-1.pdf)
- [Entregable 2](./docs/entregable-2.pdf)

## 🚀 Instalación

```bash
conda env create -f environment.yml
conda activate taa
```

## 📝 Estructura del Repositorio

Cada proyecto contiene:
- Notebooks de Jupyter con análisis exploratorio y desarrollo de modelos
- Subdirectorio `data/` con los datasets utilizados
- Scripts Python para entrenamiento y predicción
- README específico con detalles del proyecto

## 👤 Autor

Estéfano Bargas - [GitHub](https://github.com/xqft)

---

*Desarrollado como parte del curso Taller de Aprendizaje Automático, FING, Universidad de la República, 2024*
