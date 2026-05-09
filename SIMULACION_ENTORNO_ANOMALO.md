# Simulación de Entorno Anómalo – Hito 5

Este documento describe cómo simular situaciones anómalas para activar el sistema de alertas implementado en el proyecto.

---

## ✅ Objetivo

Cumplir el requisito del Hito 5:

> “Simulación de entorno anómalo”

Se simularán tres escenarios:

1. Alta latencia (alerta operativa)
2. Bajo rendimiento del modelo (alerta de modelo)
3. Data drift significativo

---

# 1️⃣ Simulación de Alta Latencia

## 🎯 Objetivo
Activar la alerta `HighAPILatency`.

## 🔧 Método

Modificar temporalmente `api.py` añadiendo:

```python
import time
time.sleep(3)
```

justo antes del `return response`.

Esto fuerza latencia > 2 segundos.

## ✅ Resultado esperado

- Prometheus detecta p95 > 2s
- Se activa alerta `HighAPILatency`
- Visible en:
  - http://localhost:9090
  - http://localhost:9093

---

# 2️⃣ Simulación de Bajo F1

## 🎯 Objetivo
Activar alerta `LowModelF1`.

## 🔧 Método

Modificar temporalmente `model_metrics.py`:

```python
MODEL_F1.set(0.4)
```

o entrenar modelo con datos corruptos.

## ✅ Resultado esperado

- model_f1_score < 0.7
- Alerta `LowModelF1` activada

---

# 3️⃣ Simulación de Data Drift

## 🎯 Objetivo
Activar alerta `DataDriftDetected`.

## 🔧 Método

Enviar requests con valores extremos:

```json
{
  "age": 5,
  "heart_rate_mean": 200,
  "sysbp_mean": 300,
  "diasbp_mean": 5,
  "resp_rate_mean": 50,
  "temperature_mean": 42,
  "spo2_mean": 50,
  "glucose_mean": 600
}
```

Repetir hasta superar DRIFT_THRESHOLD (50 requests).

## ✅ Resultado esperado

- data_drift_score > 0.5
- Alerta `DataDriftDetected`

---

# 📸 Evidencias necesarias

Capturas recomendadas:

- Dashboard Grafana con CPU/RAM
- Panel de alertas Prometheus
- Alertmanager mostrando alerta activa
- Logs de reentrenamiento si aplica

---

# ✅ Conclusión

El sistema responde correctamente ante:

- Fallos operativos
- Degradación del modelo
- Cambios en distribución de datos

Cumple el requisito de simulación de entorno anómalo del Hito 5.
