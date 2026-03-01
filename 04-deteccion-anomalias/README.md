# Detección de Anomalías

**Taller 4 - Aprendizaje No Supervisado**

## Descripción

Implementación de algoritmos de detección de anomalías para identificar patrones inusuales en datasets. Este taller explora técnicas de aprendizaje no supervisado y semi-supervisado para encontrar outliers y comportamientos anómalos.

## Objetivo

Desarrollar modelos capaces de identificar observaciones que se desvían significativamente del comportamiento normal del sistema, útil en:
- Detección de fraude
- Monitoreo de sistemas
- Control de calidad
- Ciberseguridad

## Dataset

El dataset utilizado contiene observaciones normales y anómalas, con el desafío de identificar las anomalías sin etiquetas previas (o con etiquetas limitadas para validación).

Archivos en `data/`:
- `corrected (1).gz` - Dataset procesado

## Técnicas Implementadas

### 1. Métodos Estadísticos
- Detección basada en desviación estándar
- Z-score para identificar outliers
- IQR (Interquartile Range)

### 2. Métodos de Machine Learning
- **Isolation Forest** - Aísla observaciones anómalas
- **One-Class SVM** - Clasificador de una sola clase
- **Autoencoders** - Redes neuronales para reconstrucción
- **Local Outlier Factor (LOF)** - Densidad local

### 3. Métodos de Clustering
- K-means con análisis de distancias
- DBSCAN para detección de ruido

## Métricas de Evaluación

- **Precision:** De las predichas como anomalías, cuántas realmente lo son
- **Recall:** De todas las anomalías reales, cuántas detectamos
- **F1-Score:** Balance entre precision y recall
- **ROC-AUC:** Capacidad de discriminación del modelo

## Visualizaciones

- `aciertos2.png` - Análisis de aciertos del modelo

## Desafíos

1. **Desbalance de clases:** Anomalías son raras por naturaleza
2. **Definición de "normal":** Varía según contexto y dominio
3. **Threshold selection:** Qué tan estricto ser al marcar anomalías
4. **Drift temporal:** Lo normal hoy puede ser anómalo mañana

## Archivos

- `taller4_anomalias.ipynb` - Notebook con implementaciones
- `data/` - Datasets de prueba
- `.ipynb_checkpoints/` - Checkpoints de Jupyter

## Ejecutar el Proyecto

```bash
conda activate taa
jupyter notebook taller4_anomalias.ipynb
```

## Aprendizajes Clave

- **No hay "mejor" algoritmo universal** - La efectividad depende del tipo de datos
- **Interpretabilidad vs Performance** - Métodos simples son más explicables
- **Validación es compleja** - Sin ground truth completo, es difícil medir rendimiento
- **Feature scaling crítico** - Especialmente para métodos basados en distancia
- **Domain knowledge esencial** - Entender qué es "anómalo" requiere conocimiento del dominio

## Aplicaciones Prácticas

- **Fraude financiero:** Transacciones inusuales
- **Mantenimiento predictivo:** Detectar fallos antes de que ocurran
- **Seguridad:** Intrusiones en redes
- **Salud:** Detectar condiciones médicas raras
- **Calidad:** Identificar productos defectuosos

## Comparación de Métodos

| Método | Velocidad | Interpretabilidad | Escalabilidad | Mejor para |
|--------|-----------|-------------------|---------------|------------|
| Z-score | Muy rápido | Alta | Excelente | Datos univariados |
| Isolation Forest | Rápido | Media | Buena | Datos de alta dimensión |
| One-Class SVM | Medio | Baja | Media | Fronteras complejas |
| Autoencoder | Lento | Baja | Buena | Datos complejos/imágenes |
| LOF | Lento | Media | Limitada | Clusters densos |

## Extensiones Futuras

- Deep learning para anomalías en series temporales
- Detección de anomalías colectivas (no solo puntuales)
- Sistemas en tiempo real con online learning
- Explicabilidad de detecciones (LIME, SHAP)
