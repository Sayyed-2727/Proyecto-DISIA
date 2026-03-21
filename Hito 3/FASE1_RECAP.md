# Hito 3 — Fase 1: Clarificación del problema y recap de Hito 1–2

## 1) Descripción de la fase (qué se hizo)
En esta fase se consolidó la definición operativa de **MIMIC-TRIAGE** para asegurar que el Hito 3 mantenga continuidad metodológica con Hito 1–2. Se revisaron el documento `HITO2_TEXT.md` y el notebook `HITO2_MIMIC_TRIAGE.ipynb` para fijar, sin ambigüedades, el problema clínico-técnico, la formulación de entrada/salida del modelo y el conjunto de métricas que gobernarán la comparación entre modelos.

Además, se verificaron los resultados del baseline ya implementado en Hito 2 (modelo, pipeline y rendimiento), de modo que Hito 3 parta de una referencia cuantitativa explícita y reproducible.

## 2) Resultado de la fase

### 2.1 Problema MIMIC-TRIAGE (resumen 1–2 párrafos)
**MIMIC-TRIAGE** aborda la priorización temprana de pacientes de UCI mediante predicción de **mortalidad intrahospitalaria** usando únicamente información disponible en las primeras **0–48 horas** desde el ingreso. El objetivo no es solo clasificar (vive/fallece), sino producir un **score probabilístico de riesgo** por estancia, clínicamente interpretable como intensidad de alerta temprana.

La salida probabilística se transforma en un **ranking descendente de riesgo** para soportar decisiones de triaje: los pacientes con mayor probabilidad estimada de mortalidad ocupan posiciones prioritarias para vigilancia/intervención. Este encuadre combina una evaluación estadística clásica (discriminación y calibración) con una evaluación operativa enfocada en capacidad de captura de casos de alto riesgo en el top del ranking.

### 2.2 Baseline de Hito 2 (resumen 1–2 párrafos)
El baseline de Hito 2 se implementó como un **modelo lineal probabilístico tipo regresión logística** entrenado por descenso de gradiente (sigmoide sobre combinación lineal), usando una selección de variables agregadas 0–48h y estáticas. El pipeline aplicado fue: split reproducible por `RecordID % 5 == 0` (test), imputación por mediana aprendida en train, estandarización z-score para variables numéricas y generación de probabilidades sobre test para construir el ranking clínico.

Las variables empleadas en ese baseline incluyeron estáticas (`Age_first`, `Gender_first`, `ICUType_first`) y fisiológicas agregadas (`GCS_mean`, `HR_mean`, `MAP_mean`, `SysABP_mean`, `DiasABP_mean`, `RespRate_mean`, `Temp_mean`, `Creatinine_mean`, `BUN_mean`, `WBC_mean`, `Urine_mean`). El rendimiento reportado en Hito 2 fue: **AUC-ROC = 0.7806**, **Brier = 0.0887**, con buen enriquecimiento de mortalidad en top-k.

### 2.3 Especificación cerrada para Hito 3

#### Input del modelo
Vector tabular por estancia UCI en ventana 0–48h:
- Variables estáticas (edad, género, peso, altura, tipo de UCI).
- Features agregadas de series temporales (p. ej., `first`, `last`, `mean`, `min`, `max`, `std`, `n_mediciones`, `flag_medido` según variable).

#### Output del modelo
- `p_hat(i)`: probabilidad estimada de mortalidad intrahospitalaria para la estancia `i`.
- Ranking clínico: ordenamiento descendente por `p_hat(i)` para priorización.

#### Métricas de comparación en Hito 3 (lista cerrada)
- **AUC-ROC**
- **Brier score**
- **Recall@k** para `k ∈ {25, 50, 100, 200}`
- **% fallecidos en top-k** para `k ∈ {25, 50, 100, 200}`

### 2.4 Tabla recap del baseline (Hito 2)

| Métrica | Valor |
|---|---:|
| AUC-ROC | 0.7806 |
| Brier score | 0.0887 |
| Recall@25 | 0.1684 |
| Recall@50 | 0.2421 |
| Recall@100 | 0.4526 |
| Recall@200 | 0.6421 |
| % fallecidos en top-25 | 0.6400 |
| % fallecidos en top-50 | 0.4600 |
| % fallecidos en top-100 | 0.4300 |
| % fallecidos en top-200 | 0.3050 |

## 3) Checklist de objetivos (Fase 1)
- [x] Problema clínico y técnico descrito claramente.
- [x] Baseline del Hito 2 resumido con sus métricas clave.
- [x] Lista cerrada de métricas que se usarán en la comparación.

## 4) Entregables de salida de esta fase
- Texto de clarificación del problema y recap Hito 1–2 (listo para integrar en introducción/metodología del paper).
- Tabla de métricas baseline de referencia para comparación en Fase 5.
- Definición cerrada de input/output/métricas para el diseño experimental del Hito 3.
