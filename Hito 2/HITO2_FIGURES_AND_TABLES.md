# HITO 2 - Figuras y Tablas (indice de insercion)

## Figuras
| Numero | Ruta archivo | Caption propuesto | Seccion |
|---|---|---|---|
| Figura 2.1 | `./reports/figures/label_imbalance_setA.png` | Distribucion de la etiqueta de mortalidad intrahospitalaria en Set A. | 2.4 |
| Figura 2.2 | `./reports/figures/temporal_coverage_setA_0_48h.png` | Cobertura temporal de mediciones por hora desde ingreso (0-48h). | 2.4 |
| Figura 2.3 | `./reports/figures/missingness_setA_0_48h.png` | Missingness por variable en Set A (porcentaje sin medicion en 0-48h). | 2.4 |
| Figura 2.4 | `./reports/figures/static_distributions_setA.png` | Distribuciones de variables estaticas: edad, genero y tipo de UCI. | 2.4 |
| Figura 2.5 | `./reports/figures/measurement_volume_top20_setA_0_48h.png` | Top-20 variables por volumen de mediciones en 0-48h. | 2.4 |
| Figura 2.6 | `./reports/figures/correlation_subset_setA_0_48h.png` | Matriz de correlacion para un subconjunto de variables medias en 0-48h. | 2.4 |
| Figura 2.7 | `./reports/figures/hr_mean_by_outcome_setA.png` | Comparacion de HR_mean por etiqueta de mortalidad. | 2.4 |

## Tablas
| Numero | Ruta/Origen | Caption propuesto | Seccion |
|---|---|---|---|
| Tabla 2.1 | Incluida en `HITO2_TEXT.md` | Diccionario resumido de variables clinicas relevantes en Set A (raw 0-48h). | 2.3 |
| Tabla 2.2 | Incluida en `HITO2_TEXT.md` | Esquema de ingenieria de variables por estancia en ventana 0-48h (`processed_features_48h_setA.csv`). | 2.3 |
| Tabla 2.3 | Incluida en `HITO2_TEXT.md` | Inventario cuantitativo de datos usados en Hito 2 y decision de reutilizacion de CSVs existentes. | 2.3 |
| Tabla 2.4 | Incluida en `HITO2_TEXT.md` | Resultado baseline de scoring probabilistico y ranking (split reproducible por `RecordID % 5 == 0`). | 2.5 |

## Figuras a generar
- No faltan figuras clave para cumplir el Hito 2 con los requisitos actuales.
- Figura opcional para Hito 3: `roc_curve_baseline_setA.png` (si se quiere anadir curva ROC explicita al informe).
- Figura opcional para Hito 3: `calibration_curve_baseline_setA.png` (si se quiere reforzar analisis de calibracion).
