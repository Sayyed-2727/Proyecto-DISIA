# Capitulo 7. Interpretabilidad y análisis de importancia de variables (Fase 6)

## 7.1 Descripción de la fase (qué se hizo)
En esta fase se interpretaron los modelos principales de Hito 3 sobre la corrida definitiva (`seed=42`, `n_iter=30`), tomando como referencia el baseline replicado de Hito 2 y los mejores modelos de Hito 3. Se generaron artefactos cuantitativos de interpretabilidad para:

1. Regresión logística (coeficientes estandarizados).
2. Random Forest (importancia de variables por `feature_importances_`).
3. XGBoost/GBM (estado en entorno actual: no disponible, se documenta análisis conceptual y plan).

Artefactos generados:
- `Hito 3/artifacts/tables/table_7_1_logistic_coefficients.csv`
- `Hito 3/artifacts/tables/table_7_2_random_forest_importance.csv`
- `Hito 3/artifacts/interpretability_manifest.json`

## 7.2 Regresión logística: coeficientes e interpretación

Los coeficientes se calcularon sobre variables estandarizadas (z-score), por lo que su magnitud absoluta es comparable entre predictores. Signo positivo implica aumento de riesgo; signo negativo, reducción de riesgo estimado (manteniendo el resto constante).

### 7.2.1 Top coeficientes absolutos

| Feature | Coef. estandarizado | Interpretación resumida |
|---|---:|---|
| `GCS_mean` | -0.6582 | Mayor Glasgow medio se asocia con menor riesgo (protectivo). |
| `BUN_mean` | +0.4180 | Mayor BUN medio se asocia con mayor riesgo. |
| `Age_first` | +0.3513 | Mayor edad al ingreso incrementa riesgo estimado. |
| `HR_mean` | +0.2343 | Mayor FC media se asocia con mayor riesgo. |
| `Temp_mean` | -0.1816 | Efecto protector moderado en la parametrización final. |
| `ICUType_first` | +0.1659 | Diferencias de riesgo base por cohorte de UCI. |

Notas:
- `MAP_mean` y `DiasABP_mean` quedaron con coeficiente 0 en esta solución regularizada (`L1`), lo que sugiere efecto marginal menor una vez consideradas otras variables correlacionadas.
- La señal dominante del modelo lineal queda concentrada en estado neurológico (`GCS_mean`), función renal (`BUN_mean`), edad y carga hemodinámica/respiratoria básica.

## 7.3 Random Forest: importancia de variables

### 7.3.1 Top variables por importancia

| Feature | Importance |
|---|---:|
| `GCS_last` | 0.0925 |
| `GCS_mean` | 0.0279 |
| `GCS_max` | 0.0263 |
| `Urine_mean` | 0.0210 |
| `BUN_last` | 0.0198 |
| `BUN_min` | 0.0158 |
| `HCO3_mean` | 0.0155 |
| `BUN_mean` | 0.0149 |
| `Platelets_std` | 0.0115 |
| `WBC_last` | 0.0106 |

Interpretación:
- Random Forest enfatiza múltiples aspectos del estado neurológico (`GCS_last/mean/max`), coherente con la importancia de `GCS_mean` vista en logística.
- También destaca marcadores renales/metabólicos (`BUN_*`, `Urine_mean`, `HCO3_mean`) y hematológicos (`Platelets_std`, `WBC_last`).
- La importancia se distribuye en más variables que en logística, consistente con la capacidad no lineal del modelo para capturar interacciones de mayor orden.

## 7.4 Relación con hallazgos EDA de Hito 2

La interpretación es consistente con el EDA de Hito 2 en puntos clave:

- En Hito 2 se observó desplazamiento de `HR_mean` entre fallecidos y no fallecidos; en Hito 3, `HR_mean` aparece con coeficiente positivo en logística, reforzando su utilidad predictiva.
- El enfoque por variables agregadas 0-48h se confirma útil: tanto modelos lineales como de árboles extraen señal clínica de agregados simples (`mean`, `last`, `min`, `std`) sin requerir modelado secuencial complejo.
- La ausencia de colinealidad perfecta reportada en Hito 2 permite un baseline lineal robusto; al mismo tiempo, Random Forest captura estructura adicional no lineal, explicando su rendimiento competitivo en top-k.

## 7.5 SHAP (mención conceptual y plan de extensión)

SHAP se mantiene como enfoque recomendado para explicabilidad local y global en modelos complejos (árboles/boosting), porque permite:

1. Descomponer la predicción individual en contribuciones por variable.
2. Comparar consistencia de efectos entre pacientes (heterogeneidad clínica).
3. Complementar importancias globales con evidencia local interpretable por caso.

En este entorno no se ejecutó SHAP para XGBoost porque la librería no está disponible actualmente; como extensión directa, se puede incorporar una sección SHAP en cuanto se habilite `xgboost`/`shap` en el entorno de ejecución.

## 7.6 Checklist Fase 6
- [x] Análisis de coeficientes de la regresión logística.
- [x] Importancia de variables para RF y su interpretación.
- [x] Relación de variables importantes con resultados EDA de Hito 2.
- [x] Mención razonada de técnicas tipo SHAP.
- [x] Estado de XGBoost/GBM documentado (no disponible en entorno actual).
