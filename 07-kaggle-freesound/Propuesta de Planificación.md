## Pendientes
- 80 clases
- Labeling multiclass

### Introducción: 
- [ ] Leer a detalle overview del proyecto
- [ ] Estudiar la métrica ( label-weighted label-ranking average precision )
- [ ] Agregar clases y funciones que ya tengamos para el manejo o visualización de datos.

### Exploración de datos:
- [ ] Verificar imágenes y etiquetas.
- [ ] Distribución de los datos.
- [ ] Metadatos.
- [ ] Visualizar ejemplos.

### Preprocesamiento:
- [ ] Aplicar resizing.
- [ ] Normalizar.
- [ ] Data augmentation (aumentar o disminuir velocidad, agregar ruido, grave o agudo, volumen).
- [ ] Cleaning

### Selección del modelo:
- [ ] Elegir varias arquitecturas (CNNs)

### Transfer learning: 
- [ ] Investigar las diferentes arquitecturas y ver cuáles se adaptan mejor a nuestros datos.
- [ ] VGG16: It's good for image classification but might not capture very fine details due to its simplicity.
- [ ] ResNet: The ResNet architecture introduced the concept of residual connections or skip connections, which help in training very deep networks. ResNet models, especially ResNet50 and ResNet101, are powerful for image classification tasks and can capture more complex patterns compared to VGG16.
- [ ] EfficientNet:  This architecture scales the network's depth, width, and resolution in a balanced manner. EfficientNet is known for being both computationally efficient and highly accurate.
- [ ] Cambiar el uso de las arquitecturas según la tabla de referencia

  ![image](https://github.com/ValenAlaniz/Laboratorio-2-TAA/assets/87722864/41012432-b0b6-4689-a31c-d351dde6b651)

### Entrenamiento y validación:
- [ ] Separar los datos en validación y entrenamiento con un porcentaje razonable.
- [ ] Verificar overfitting graficando métricas en train y validación.
- [ ] Controlar sobreajuste con técnicas de regularización (Dropout, batchnormalization para exploción/desvanecimiento de gradientes, EarlyStopping)

### Tunneo de hiperparametros
- [ ] Optuna.
- [ ] Keras tunner.

### Documentación
- [ ] Documentación del Proyecto en [Overleaf](https://www.overleaf.com/). Ir escribiendo el informe durante todo el proceso.

### Preguntas de la Propuesta de los profes: 
- [ ] ...

**Notas:**
- Al aplicar algo mostrar la mejora y pensar por qué tiene sentido hacerlo, por ejemplo: si hacemos data augmentation mostrar en una tabla como es el rendimiento con y sin el aumento
- No olvidarnos de cargar todos los experimentos en comet. 


### Configuraciones previas: 
- [ ] Crear un proyecto con Comet.
- [ ] Configurar una instancia de AWS.
- [ ] [Overleaf](https://www.overleaf.com/).

### **Instancia AWS**
- GPU: p2.xlarge, p3.2xlarge o g4dn.xlarge
- Storage: Por lo menos 100Gb de SSD
-  RAM: El conjunto de entrenamiento pesa ≈35Gb, si queremos tener todo eso en memoria nos alcanza con una p3.2xlarge que tiene 61Gb de RAM.
-  AMI: AWS Deep Learning AMI (Ubuntu 18.04) Version 34.0 or later
