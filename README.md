# Machine Learning Workshop (TAA)

**Faculty of Engineering, Universidad de la Republica**
**2024 Course**

This repository consolidates the individual projects developed during the Machine Learning Workshop course, based on the book "Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow" by Aurelien Geron.

## Repository Structure

### Weekly Workshops

Five hands-on workshops developed throughout the semester, each exploring different machine learning techniques:

- **[01-clasificador-titanic](./01-clasificador-titanic/)** - Binary classifier for Titanic survival prediction
- **[02-sentiment-imdb](./02-sentiment-imdb/)** - Sentiment analysis on movie reviews
- **[03-regresion-bicicletas](./03-regresion-bicicletas/)** - Bike demand prediction (Random Forest + XGBoost)
- **[04-deteccion-anomalias](./04-deteccion-anomalias/)** - Anomaly detection on datasets
- **[05-bicicletas-rnn](./05-bicicletas-rnn/)** - Time-series prediction with recurrent neural networks

### Kaggle Projects

Two competitive projects developed during the course, with full code and documentation:

6. **[06-kaggle-higgs-boson](./06-kaggle-higgs-boson/)** - Particle physics event classification ([Report](./docs/higgs-boson-informe.pdf))
7. **[07-kaggle-freesound](./07-kaggle-freesound/)** - Audio classification with deep learning ([Report](./docs/freesound-informe.pdf))

### Documentation

The `docs/` folder contains PDF reports for all projects:

- `higgs-boson-informe.pdf` - Project 1: Higgs Boson Challenge
- `freesound-informe.pdf` - Project 2: Freesound Audio Tagging
- `entregable-1.pdf` - First project deliverable
- `entregable-2.pdf` - Second project deliverable

## Key Results

- **Sentiment Analysis (IMDB):** 87% accuracy with TF-IDF, bigrams, and stopword removal
- **Bike Demand (XGBoost):** RMSLE ~0.37 on cross-validation (Kaggle top 5% ~0.35)
- **Titanic Classifier:** Identified social class and gender as critical survival variables

## Environment Setup

```bash
# Create conda environment with all dependencies
conda env create -f environment.yml
conda activate taa

# Launch Jupyter to explore notebooks
jupyter notebook
```

## Tech Stack

- **Frameworks:** scikit-learn, Keras, TensorFlow, PyTorch
- **Data processing:** pandas, numpy, matplotlib, seaborn
- **Models:** Random Forest, XGBoost, RNN/LSTM
- **Techniques:** TF-IDF, SHAP values, cross-validation
