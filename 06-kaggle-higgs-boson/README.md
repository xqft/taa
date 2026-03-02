# Proyecto Kaggle: Higgs Boson Challenge

Este laboratorio forma parte del taller de Aprendizaje Automático y se enfoca en el análisis del conjunto de datos proporcionado en la competencia del [Bosón de Higgs en Kaggle](https://www.kaggle.com/c/higgs-boson).

## Estructura del Proyecto

### Notebooks de Modelos Finales (`notebooks/modelos-finales/`)

- **AnalisisDeDatos.ipynb** - Análisis exploratorio de datos (EDA) principal
- **XGBoost.ipynb** - Mejor modelo tradicional usando XGBoost
- **red_neuronal.ipynb** - Modelo de deep learning
- **mejor_modelo.ipynb** - Modelo final consolidado con mejores hiperparámetros

### Notebooks de Exploración (`notebooks/exploracion/`)

Notebooks experimentales desarrollados durante el proceso de exploración:
- **AdaBoost.ipynb**, **ExtraTrees.ipynb**, **RandomForests.ipynb** - Experimentos con modelos de ensemble
- **EDA.ipynb**, **Notebook.ipynb** - Análisis exploratorios iniciales
- **download_data.ipynb** - Script de descarga de datos
- **Oversampling-RegrecionLogistica.ipynb** - Experimentos con oversampling
- **pruebas_threshold.ipynb**, **pruebas_threshold-oversampling.ipynb** - Optimización de umbral de decisión
- **Untitled.ipynb** - Notebook de pruebas

## Configuración Inicial

Antes de comenzar es necesario estar registrado en [Kaggle](https://www.kaggle.com) y haber aceptado las condiciones de la competencia.

### Instalación de Kaggle API

Configuración de la API de Kaggle:

1. Instalar el paquete de Kaggle API utilizando pip:

   ```sh
    pip install kaggle
   ```

2. En tu cuenta de kaggle descarga una key en la sección de API en la página de configuración.

3. Mover el archivo kagle.json a la carpeta .kaggle en el directorio home:

   ```sh
   mkdir -p ~/.kaggle/ && mv kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
   ```

4. Descargar el conjunto de entrenamiento y descomprimir todos los archivos:

   ```sh
    unzip higgs-boson.zip
    unzip test.zip
    unzip training.zip
   ```

### Intalación de requisitos

El archivo requirements.txt contiene todas las dependencias necesarias para ejecutar el proyecto. Instalarlas usando:

```sh
 pip install -r requirements.txt
```
Como alternativa, se puede utilizar Conda:
```bash
conda env create -f conda.yml
```
