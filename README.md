# Proyecto DISIA – MIMIC-TRIAGE  
## Sistema Integral de Predicción de Mortalidad y MLOps en Entorno UCI

Este repositorio recoge el desarrollo completo del proyecto **MIMIC-TRIAGE**, cuyo objetivo es la estimación del riesgo de mortalidad intrahospitalaria en pacientes de UCI y la construcción de un sistema MLOps profesional con monitorización, versionado y feedback loop automático.

El proyecto ha sido desarrollado siguiendo los cinco hitos de la asignatura **Desarrollo e Integración de Servicios de Inteligencia Artificial (DISIA)**, evolucionando desde la definición conceptual hasta un sistema industrial completo.

---

# 📌 Visión General del Proyecto

El sistema permite:

- Estimar la probabilidad de mortalidad hospitalaria a partir de datos clínicos de las primeras 48h en UCI.
- Generar priorización clínica basada en riesgo.
- Desplegar el modelo en un entorno realista.
- Monitorizar su comportamiento en producción.
- Detectar deriva de datos.
- Reentrenar automáticamente el modelo si es necesario.
- Versionar y promover modelos mediante MLflow Model Registry.

Se ha trabajado con una adaptación del dataset **PhysioNet/CinC 2012**, generando una representación tabular por paciente.

---

# 🏗 Evolución por Hitos

---

# ✅ Hito 1 – Definición del Problema

En esta fase se establecieron:

- Definición del problema clínico.
- Justificación del uso de IA en priorización UCI.
- Objetivos del sistema.
- Métricas de evaluación.
- Alcance técnico del proyecto.

Se definió el objetivo principal:

> Estimar riesgo de mortalidad intrahospitalaria en base a datos clínicos estructurados.

---

# ✅ Hito 2 – Análisis Exploratorio y Preparación de Datos

Se realizó:

- Exploración detallada del dataset MIMIC / PhysioNet.
- Limpieza y tratamiento de valores faltantes.
- Análisis de distribución de variables.
- Estudio de correlaciones.
- Construcción de features agregadas (media, máximo, mínimo, etc.).
- Generación del dataset final tabular.

Resultados:

- Dataset estructurado por paciente.
- Variables clínicas relevantes seleccionadas.
- Análisis de balance de clases.

---

# ✅ Hito 3 – Modelado y Validación

En esta fase se desarrolló el pipeline de modelado:

### Modelos evaluados:

- Logistic Regression (baseline)
- Random Forest
- Gradient Boosting
- Modelos ajustados con GridSearch

### Técnicas aplicadas:

- Validación cruzada
- Ajuste de hiperparámetros
- Comparación de métricas (AUC, F1, Accuracy)
- Interpretabilidad mediante SHAP
- Análisis de importancia de variables

Resultado:

- Selección de un modelo óptimo.
- Validación experimental sólida.
- Análisis de interpretabilidad clínica.

---

# ✅ Hito 4 – Despliegue

Se implementó:

- API REST con **FastAPI**.
- Endpoint `/predict`.
- Integración del modelo entrenado.
- Contenerización con Docker.
- Arquitectura basada en servicios.

El sistema quedó desplegado como servicio de inferencia en tiempo real.

---

# ✅ Hito 5 – Monitorización y Feedback Loop (MLOps)

En esta fase el sistema evolucionó a un entorno profesional MLOps.

---

## 🔎 Monitorización Operativa

Se integró:

- Prometheus para recolección de métricas.
- Grafana para visualización.
- Métricas como:
  - Requests per second
  - Latencia media y p95
  - Error rate
  - CPU y RAM usage

---

## 🤖 Monitorización del Modelo

Se añadieron métricas específicas de modelo:

- Prediction rate
- Average prediction confidence
- Champion vs Challenger traffic
- Model version served
- Drift detection (PSI)

---

## 📊 Data Drift Detection

Se implementó:

- Cálculo de **Population Stability Index (PSI)**.
- Comparación distribución entrenamiento vs producción.
- Umbral de alerta si PSI > 0.25.

Esto permite detectar cambios en el comportamiento de los datos.

---

## 🚨 Sistema de Alertas

Configurado con Prometheus + Alertmanager:

- Alta latencia
- Alto error rate
- API caída
- Deriva de datos

---

## 🏆 Champion / Challenger

Estrategia A/B implementada:

- 80% tráfico → Champion
- 20% tráfico → Challenger

Permite evaluar nuevos modelos en producción minimizando riesgo.

---

## 🔁 Feedback Loop Automático

Cuando:

- Se detecta drift significativo
- Se degrada el rendimiento

El sistema:

1. Ejecuta reentrenamiento.
2. Evalúa métricas.
3. Registra nueva versión en MLflow.
4. Promueve automáticamente a Production.

---

# 🧠 Tecnologías Utilizadas

- Python
- FastAPI
- Scikit-learn
- MLflow
- Prometheus
- Grafana
- Docker / Docker Compose
- Pydantic
- SHAP

---

# 📂 Estructura del Repositorio

```
codigo/
  src/
  models/
  data/
monitoring/
dashboard/
mlflow-data/
docker-compose.yml
README.md
```

---

# 🚀 Estado Actual del Proyecto

El sistema actual representa una arquitectura MLOps completa:

- Modelo entrenado y validado
- Despliegue funcional
- Monitorización avanzada
- Alertas activas
- Drift detection
- Versionado formal
- Reentrenamiento automático

El proyecto no es únicamente un modelo predictivo, sino un sistema IA productivo completo.

---

# 📈 Posibles Mejoras Futuras

- Shadow deployment
- Monitorización de fairness
- Integración CI/CD real
- Kubernetes deployment
- Integración con bases de datos clínicas reales

---

# 🎓 Contexto Académico

Proyecto desarrollado para la asignatura:

**Desarrollo e Integración de Servicios de Inteligencia Artificial (DISIA)**

Máster en Ingeniería Informática.

---

# 📌 Conclusión

Este repositorio documenta la evolución desde un análisis exploratorio hasta un sistema MLOps industrial con monitorización, alertas, versionado y feedback loop automático.

El proyecto cumple tanto objetivos académicos como estándares técnicos reales de producción.
