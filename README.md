# Proyecto DISIA - MIMIC-TRIAGE

Este repositorio recoge el desarrollo del proyecto **MIMIC-TRIAGE**, centrado en la estimación de riesgo de mortalidad intrahospitalaria y en la generación de rankings clínicos de priorización a partir de datos de UCI.

El trabajo se ha estructurado siguiendo los hitos de la asignatura:

- **Hito 1:** definición del problema, objetivos, contexto y planteamiento general del proyecto.
- **Hito 2:** análisis exploratorio de datos, preparación del dataset y formulación del problema de predicción y ranking.
- **Hito 3:** modelado, experimentación y validación de distintos modelos de aprendizaje automático, incluyendo baseline, comparación de modelos, validación cruzada, ajuste de hiperparámetros e interpretabilidad.
- **Hito 4:** despliegue e integración de la solución seleccionada en un entorno de uso más cercano a un sistema real.

El proyecto utiliza una adaptación del dataset **PhysioNet/CinC 2012**, trabajando sobre información clínica de las primeras 48 horas de estancia en UCI. A partir de estas mediciones se construye una representación tabular por paciente que permite entrenar y evaluar modelos de riesgo.

## Estado actual

Actualmente, el proyecto cuenta con una fase de modelado validada experimentalmente, donde se han comparado varios enfoques y se ha seleccionado un modelo principal para continuar con la fase de despliegue.

## Nota

La explicación detallada del trabajo realizado en cada fase se encuentra en la documentación y memoria asociadas a cada hito.