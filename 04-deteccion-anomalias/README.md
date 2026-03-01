# Detección de Anomalías

## Descripción

Aplicación de técnicas de aprendizaje no supervisado para identificar patrones anómalos en conjuntos de datos. Este proyecto explora diferentes enfoques para detectar observaciones que se desvían significativamente del comportamiento normal.

## Dataset

Datos con características que presentan patrones normales y anómalos a identificar.

## Metodología

### Técnicas de Detección
Implementación y comparación de múltiples enfoques:
- **Métodos estadísticos**: Detección basada en desviaciones estadísticas
- **Métodos basados en densidad**: Identificación de regiones de baja densidad
- **Métodos de aislamiento**: Isolation Forest y variantes
- **Autoencoders**: Detección mediante reconstrucción

### Evaluación
- Análisis de falsos positivos y falsos negativos
- Métricas de precisión en la detección de anomalías
- Visualización de resultados

## Resultados

El proyecto incluye:
- Comparación de diferentes técnicas de detección
- Análisis de aciertos y errores
- Visualizaciones de los patrones detectados (`aciertos2.png`)
- Evaluación de la efectividad de cada método

## Aplicaciones

La detección de anomalías tiene aplicaciones en:
- Detección de fraude
- Monitoreo de sistemas
- Control de calidad
- Seguridad informática
- Detección de fallas en equipamiento

## Tecnologías

- **scikit-learn**: Implementación de algoritmos de detección
- **NumPy/pandas**: Procesamiento de datos
- **matplotlib**: Visualización de anomalías
- Posiblemente **TensorFlow/Keras**: Para autoencoders

## Archivos

- `taller4_anomalias.ipynb`: Notebook con análisis completo
- `data/corrected.gz`, `data/corrected (1).gz`: Datasets comprimidos
- `*.png`: Visualizaciones de resultados

## Aprendizajes Clave

- Diferentes enfoques para detección de anomalías
- Trade-offs entre sensibilidad y especificidad
- Importancia de la visualización en aprendizaje no supervisado
- Aplicaciones prácticas de detección de outliers

## Ejecución

```bash
# Desde el directorio del proyecto
jupyter notebook taller4_anomalias.ipynb
```

---

[← Volver al repositorio principal](../)
