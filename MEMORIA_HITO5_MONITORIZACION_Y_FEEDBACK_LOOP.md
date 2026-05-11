# HITO 5 – Monitorización y Feedback Loop  
Proyecto DISIA – Desarrollo e Integración de Servicios de Inteligencia Artificial  

---

# 1. Introducción

En este hito se implementa un sistema completo de **monitorización, alertas y feedback loop** para un servicio de predicción de mortalidad hospitalaria desplegado en producción.

El objetivo es garantizar:

- Observabilidad a nivel de infraestructura, aplicación y modelo.
- Detección temprana de degradación del rendimiento.
- Activación automática de alertas.
- Capacidad de reentrenamiento y despliegue controlado de nuevos modelos.

Este diseño sigue los principios descritos en:

- Hito 5.pdf  
- Tema 5 – Monitorización  
- Tema 5 – Feedback Loop  

---

# 2. Selección de tecnología de monitorización

Se seleccionó la siguiente arquitectura:

- **Prometheus** → Recolección de métricas.
- **Grafana** → Visualización.
- **Alertmanager** → Gestión de alertas.
- **Evidently AI** → Detección de data drift.
- **psutil** → Métricas de infraestructura (CPU y RAM).

## Justificación

Según el documento de Monitorización:

- Es necesario monitorizar infraestructura, modelo y aplicación.
- Se recomienda instrumentación + alertas + capacidad de depuración.
- Grafana está mencionada explícitamente como tecnología válida.

Prometheus + Grafana es una solución estándar en entornos DevOps/MLOps y permite:

- Métricas en tiempo real.
- Definición de reglas de alerta.
- Integración con múltiples sistemas.

---

# 3. Métricas implementadas

## 3.1 Métricas de infraestructura

- `system_cpu_usage_percent`
- `system_memory_usage_percent`

Permiten responder a:

- ¿El consumo de CPU es correcto?
- ¿Existe saturación de memoria?

Alineado con:
> “¿El consumo de CPU y memoria del modelo/servicio es el correcto?”

---

## 3.2 Métricas de aplicación

- `api_request_count`
- `api_request_latency_seconds`
- `model_errors_total`

Permiten detectar:

- Cuellos de botella.
- Fallos del servicio.
- Degradación del rendimiento.

---

## 3.3 Métricas del modelo

- `model_precision`
- `model_f1_score`
- `model_auc_score`
- `model_version_served_total`

Permiten evaluar:

- Degradación estadística.
- Comparación entre versiones.
- Rendimiento real en producción.

---

# 4. Detección de deriva de datos

Se implementó un sistema de detección de **input drift** usando Evidently AI.

Funcionamiento:

- Se almacena dataset de referencia (entrenamiento).
- Cada 50 requests se compara distribución actual vs histórica.
- Se calcula `data_drift_score`.
- Se exporta como métrica Prometheus.

Esto responde a:

> “¿La distribución de mis datos sigue siendo la misma?”

---

# 5. Sistema de alertas

Se definieron reglas en Prometheus:

## 5.1 Alerta operativa

- `HighAPILatency`
  - Si p95 > 2 segundos durante 1 minuto.

## 5.2 Alerta de modelo

- `LowModelF1`
  - Si F1 < 0.7.

## 5.3 Alerta de drift

- `DataDriftDetected`
  - Si drift_score > 0.5.

Alertmanager permite:

- Notificaciones por webhook.
- Extensible a email, Telegram o Teams.

Se cumple el requisito:

> “Al menos una alerta para métrica operativa y métrica de modelo”.

---

# 6. Simulación de entorno anómalo

Se diseñaron tres simulaciones:

1. Forzar latencia artificial.
2. Forzar F1 bajo.
3. Enviar datos extremos para generar drift.

Se documenta el procedimiento en:
`SIMULACION_ENTORNO_ANOMALO.md`

Esto cumple el requisito explícito de simulación.

---

# 7. Feedback Loop

## 7.1 Reentrenamiento simple

Se implementó `retrain.py` que:

- Reentrena el modelo.
- Calcula métricas.
- Compara contra el champion actual.

## 7.2 Estrategia de despliegue

Se implementó **A/B Testing (Champion / Challenger)**:

- 80% tráfico → Champion
- 20% tráfico → Challenger

Justificación (según FeedbackLoop.pdf):

A/B permite:

- Experimentación controlada.
- Comparación directa de rendimiento.
- Reducción de riesgo frente a despliegue completo.

## 7.3 Promoción automática

Si:

F1_nuevo > F1_champion

Entonces:

- Se promueve automáticamente a nuevo best_model.pkl

Esto implementa un feedback loop real basado en rendimiento.

---

# 8. Arquitectura final

Servicios desplegados:

- API FastAPI
- Servicio de entrenamiento
- Prometheus
- Grafana
- Alertmanager

El sistema permite:

- Observabilidad completa.
- Detección automática de problemas.
- Reentrenamiento controlado.
- Despliegue experimental de modelos.

---

# 9. Conclusión

El sistema desarrollado cumple todos los requisitos del Hito 5:

✔ Monitorización de infraestructura  
✔ Monitorización de aplicación  
✔ Monitorización de modelo  
✔ Detección de drift  
✔ Alertas operativas y de modelo  
✔ Simulación de entorno anómalo  
✔ Reentrenamiento simple  
✔ Bonus A/B (Champion/Challenger)  

Se ha implementado una arquitectura MLOps completa alineada con los principios de monitorización y feedback loop descritos en el temario.

El sistema es reproducible, escalable y profesional.
