# Capitulo 2. Los datos: origen, descripcion y analisis

## 2.1 Introduccion
Este capitulo describe el origen, la estructura y la calidad de los datos usados en MIMIC-TRIAGE para el Hito 2, siguiendo la plantilla del profesor. El objetivo es demostrar que la formulacion del problema de priorizacion clinica se apoya en una comprension correcta del dataset, de su unidad de analisis y de sus limitaciones.

La tarea se define como prediccion de mortalidad intrahospitalaria a nivel de estancia UCI, usando unicamente informacion disponible en las primeras 48 horas desde el ingreso. El resultado del capitulo es un dataset tabular por estancia, listo para modelado en Hito 3, junto con un analisis exploratorio orientado a decisiones de modelado y evaluacion.

El trabajo es reproducible y ejecutable en el notebook `HITO2_MIMIC_TRIAGE.ipynb`, con salida principal en `./csvs/processed_features_48h_setA.csv`.

## 2.2 Fuentes, origen y tamano de los datos
La fuente principal es el dataset del reto PhysioNet/Computing in Cardiology 2012, disponible en el repositorio local en:

`./predicting-mortality-of-icu-patients-the-physionet-computing-in-cardiology-challenge-2012-1.0.0/`

El dataset se distribuye por conjuntos (Set A, B y C), con outcomes en archivos separados (`Outcomes-a.txt`, `Outcomes-b.txt`, `Outcomes-c.txt`). En el raw local:
- `set-a`: 4000 ficheros de estancias (`*.txt`)
- `set-b`: 4000 ficheros de estancias (`*.txt`)
- `set-c`: no descomprimido como carpeta en esta copia local (si existen `set-c.zip` y `Outcomes-c.txt`)

Para Hito 2 se usa Set A (etiquetado), con verificacion explicita:
- 4000 ficheros en `set-a`
- 4000 filas en `Outcomes-a.txt`
- correspondencia 1:1 por `RecordID` (sin faltantes ni duplicados cruzados)

Formato raw por estancia:
- cada fichero representa una estancia UCI
- esquema largo con columnas `Time, Parameter, Value`
- una observacion raw = una medicion de un parametro clinico en un instante temporal

Unidad de analisis final para modelado:
- una fila por estancia (`RecordID`)
- features agregadas en ventana explicita 0-48h
- etiqueta: `In-hospital_death` (solo de outcomes, no del raw de senales)

Ventana temporal:
- se aplica filtro `Time <= 48h` de forma explicita para evitar leakage temporal
- en Set A, el maximo observado en raw es 48.0h

## 2.3 Inventario de datos
El inventario se organiza en tres niveles: raw clinico por estancia, outcomes con etiqueta y dataset procesado para modelado.

Artefactos principales del Hito 2:
- Notebook de trabajo: `HITO2_MIMIC_TRIAGE.ipynb`
- Dataset procesado para Hito 3: `./csvs/processed_features_48h_setA.csv`
- Reporte de validacion de CSVs heredados: `./csvs/csv_validation_report.csv`
- Auditoria de datos: `./csvs/hito2_data_audit_setA.json`

[TABLA 2.1 AQUI]

**Caption Tabla 2.1.** Diccionario resumido de variables clinicas relevantes en Set A (raw 0-48h).  

| Variable | Tipo | Descripcion clinica | Unidad (referencia) | Rol en pipeline |
|---|---|---|---|---|
| RecordID | Identificador | ID unico de estancia UCI | - | Clave de union |
| Age | Estatica | Edad al ingreso | years | Feature estatica |
| Gender | Estatica | Sexo codificado (0/1) | - | Feature estatica |
| ICUType | Estatica | Tipo de UCI (1-4) | - | Segmentacion/cohorte |
| Height | Estatica | Altura al ingreso | cm | Feature estatica (con faltantes codificados) |
| Weight | Estatica | Peso al ingreso | kg | Feature estatica (con faltantes codificados) |
| HR | Temporal | Frecuencia cardiaca | bpm | Feature temporal agregada |
| MAP | Temporal | Presion arterial media invasiva | mmHg | Feature temporal agregada |
| SysABP | Temporal | Presion arterial sistolica invasiva | mmHg | Feature temporal agregada |
| DiasABP | Temporal | Presion arterial diastolica invasiva | mmHg | Feature temporal agregada |
| RespRate | Temporal | Frecuencia respiratoria | breaths/min | Feature temporal agregada |
| Temp | Temporal | Temperatura corporal | C | Feature temporal agregada |
| Creatinine | Temporal | Funcion renal (creatinina) | mg/dL | Feature temporal agregada |
| BUN | Temporal | Nitrogeno ureico | mg/dL aprox. | Feature temporal agregada |
| WBC | Temporal | Leucocitos | K/uL | Feature temporal agregada |
| Urine | Temporal | Diuresis | mL | Feature temporal agregada |
| In-hospital_death | Etiqueta | Mortalidad intrahospitalaria | binaria (0/1) | Target |

[TABLA 2.2 AQUI]

**Caption Tabla 2.2.** Esquema de ingenieria de variables por estancia en ventana 0-48h (`processed_features_48h_setA.csv`).  

| Grupo de variables | Operador/feature | Definicion operacional | Ejemplo de nombre de columna | Justificacion |
|---|---|---|---|---|
| Temporales | first | Primer valor observado en 0-48h | `HR_first` | Estado inicial temprano |
| Temporales | last | Ultimo valor observado en 0-48h | `Creatinine_last` | Estado al final de ventana |
| Temporales | mean | Media en 0-48h | `MAP_mean` | Nivel promedio de severidad |
| Temporales | min | Minimo en 0-48h | `SysABP_min` | Eventos de extremo inferior |
| Temporales | max | Maximo en 0-48h | `Temp_max` | Eventos de extremo superior |
| Temporales | std | Desviacion estandar en 0-48h | `RespRate_std` | Variabilidad fisiologica |
| Temporales | n_mediciones | Numero de observaciones en 0-48h | `WBC_n_mediciones` | Intensidad de monitorizacion |
| Temporales | flag_medido | Indicador de al menos 1 medicion | `Lactate_flag_medido` | Missingness informativo |
| Estaticas | first (equivalente) | Valor unico/estable por estancia | `Age_first`, `ICUType_first` | Cohorte y ajuste de riesgo |
| Etiqueta | outcome | Mortalidad intrahospitalaria | `In-hospital_death` | Objetivo de prediccion |

[TABLA 2.3 AQUI]

**Caption Tabla 2.3.** Inventario cuantitativo de datos usados en Hito 2 y decision de reutilizacion de CSVs existentes.  

| Elemento | Tamano/estado | Uso en Hito 2 | Comentario |
|---|---|---|---|
| `set-a/*.txt` | 4000 estancias | Si | Fuente raw principal etiquetable |
| `Outcomes-a.txt` | 4000 filas | Si | Etiqueta `In-hospital_death` |
| Match IDs set-a vs outcomes-a | 1:1 | Si | Sin discrepancias |
| `processed_features_48h_setA.csv` | 4000 x 338 | Si | Dataset final para Hito 3 |
| CSVs heredados en `./csvs/` | 4 ficheros validados | No (como input de modelado) | No son matriz por estancia 0-48h; riesgo de leakage en resumentes |

### 2.3.1 Limitaciones y sesgos del dato
- **Missingness no aleatorio**: la ausencia de medicion suele depender de decision clinica, no de azar; por eso se modela `flag_medido` y `n_mediciones`.
- **Muestreo irregular**: las series tienen frecuencias heterogeneas entre variables y entre estancias; se resume por agregaciones robustas en 0-48h.
- **Outliers y unidades**: algunas variables tienen rangos extremos o unidades no uniformes entre fuentes; se requiere control de calidad en Hito 3 (winsorizacion/escala, segun modelo).
- **Riesgo de leakage**: campos como `Length_of_stay` y `Survival` no se usan como features de triage temprano.
- **Desbalanceo de clases**: mortalidad positiva ~13.85%, por lo que se deben priorizar metricas y estrategias robustas a clases desbalanceadas.

## 2.4 Analisis de los datos
El analisis exploratorio se orienta a preguntas concretas para formular correctamente el problema de ranking clinico.

### Pregunta 1: Cual es el grado de desbalanceo de la etiqueta?
[FIGURA 2.1 AQUI]

Archivo: `./reports/figures/label_imbalance_setA.png`  
Caption: **Distribucion de la etiqueta de mortalidad intrahospitalaria en Set A.**

Interpretacion:
1. La clase positiva (fallece) representa aproximadamente el 13.85% de estancias, frente al 86.15% de supervivientes.
2. El problema es desbalanceado y no debe evaluarse solo con accuracy.
3. Este patron justifica incluir metricas de ranking top-k y calibracion probabilistica.

### Pregunta 2: Como se distribuye la captura temporal de mediciones en 0-48h?
[FIGURA 2.2 AQUI]

Archivo: `./reports/figures/temporal_coverage_setA_0_48h.png`  
Caption: **Cobertura temporal de mediciones por hora desde ingreso (0-48h).**

Interpretacion:
1. Existe actividad de medicion durante toda la ventana, con variacion por hora.
2. La evidencia respalda que 0-48h contiene senal clinica suficiente para triage inicial.
3. La irregularidad temporal sugiere representar dinamica mediante resumenes por estancia.

### Pregunta 3: Que variables presentan mayor ausencia de medicion?
[FIGURA 2.3 AQUI]

Archivo: `./reports/figures/missingness_setA_0_48h.png`  
Caption: **Missingness por variable en Set A (porcentaje sin medicion en 0-48h).**

Interpretacion:
1. Variables como `TroponinI`, `Cholesterol` y `TroponinT` tienen alta ausencia, compatible con pruebas selectivas.
2. Variables basicas (`Age`, `Gender`, `ICUType`, `Weight`, `Height`) tienen cobertura completa.
3. El missingness aporta informacion clinica implita y no debe eliminarse sin modelarlo.

### Pregunta 4: Cual es el perfil de variables estaticas de la cohorte?
[FIGURA 2.4 AQUI]

Archivo: `./reports/figures/static_distributions_setA.png`  
Caption: **Distribuciones de variables estaticas: edad, genero y tipo de UCI.**

Interpretacion:
1. La cohorte incluye diversidad de edades y tipos de UCI, sin concentrarse en un unico subgrupo.
2. El tipo de UCI introduce heterogeneidad de case-mix que debe controlarse en modelado.
3. Estas variables son candidatas naturales de ajuste en cualquier baseline.

### Pregunta 5: Que variables concentran mayor volumen de monitorizacion?
[FIGURA 2.5 AQUI]

Archivo: `./reports/figures/measurement_volume_top20_setA_0_48h.png`  
Caption: **Top-20 variables por volumen de mediciones en 0-48h.**

Interpretacion:
1. Variables hemodinamicas y de laboratorio rutinario concentran la mayor parte del volumen.
2. Esto confirma que el dataset combina monitorizacion continua y analitica discreta.
3. El volumen desigual entre variables refuerza el uso de features de conteo (`n_mediciones`).

### Pregunta 6: Existen relaciones entre variables fisiologicas agregadas?
[FIGURA 2.6 AQUI]

Archivo: `./reports/figures/correlation_subset_setA_0_48h.png`  
Caption: **Matriz de correlacion para un subconjunto de variables medias en 0-48h.**

Interpretacion:
1. Se observan relaciones moderadas entre variables hemodinamicas relacionadas fisiologicamente.
2. No hay colinealidad perfecta en el subconjunto mostrado, lo que permite usar modelos lineales baseline.
3. Para Hito 3 conviene complementar con modelos no lineales y analisis de importancia.

### Pregunta 7: Cambia algun marcador fisiologico entre fallecidos y no fallecidos?
[FIGURA 2.7 AQUI]

Archivo: `./reports/figures/hr_mean_by_outcome_setA.png`  
Caption: **Comparacion de `HR_mean` por etiqueta de mortalidad.**

Interpretacion:
1. La distribucion de `HR_mean` en fallecidos aparece desplazada respecto al grupo no fallecido.
2. Aunque no implica causalidad, sugiere utilidad predictiva combinada con otras variables.
3. Este tipo de contraste apoya la formulacion de scoring probabilistico multivariable.

## 2.5 Evaluacion del rendimiento. Metricas
La salida del sistema se define como una probabilidad de mortalidad por estancia:

`p_hat(i) = P(In-hospital_death_i = 1 | x_i, ventana 0-48h)`

donde `x_i` es el vector de features agregadas de la estancia `i`.  
El ranking clinico se obtiene ordenando de mayor a menor:

`ranking = sort_desc( p_hat(i) )`

### Metricas probabilisticas
- **AUC-ROC**: mide capacidad de discriminacion global entre fallecidos y no fallecidos.
- **Brier score**: mide error cuadratico medio de probabilidades predichas (calibracion + discriminacion).

### Metricas de ranking clinico
- **Recall@k**: proporcion de fallecidos capturados en los primeros `k` pacientes del ranking.
- **% fallecidos en top-k**: pureza de la zona prioritaria del ranking (fallecidos/k).

[TABLA 2.4 AQUI]

**Caption Tabla 2.4.** Resultado baseline de scoring probabilistico y ranking (split reproducible por `RecordID % 5 == 0` para test).  

| Metrica | Valor |
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

Interpretacion de metricas:
1. El baseline ya separa riesgo de forma util (AUC ~0.78) sin usar informacion posterior a 48h.
2. En terminos operativos, el top-k concentra mortalidad por encima de la prevalencia base.
3. Para Hito 3 se debe mejorar robustez con validacion formal, calibracion y comparacion entre modelos.

### Conexion con Hito 3 (modelado)
Con este capitulo queda cerrada la formulacion de problema:
- input: features por estancia en 0-48h (`processed_features_48h_setA.csv`)
- output: probabilidad de mortalidad
- decision: ranking descendente por riesgo para priorizacion
- evaluacion: AUC/Brier + metricas top-k orientadas a triage

---

## Checklist final: "Listo para entregar?"
- [x] Estructura del capitulo exacta (2.1 a 2.5).
- [x] Origen, fuentes y tamano del dataset descritos con evidencia.
- [x] Inventario de datos con TABLA 2.1 y TABLA 2.2 incluidas.
- [x] Limitaciones y sesgos del dato explicitados.
- [x] Analisis EDA orientado por preguntas (no solo graficos).
- [x] Todas las figuras con marcador, ruta, caption e interpretacion.
- [x] Formulacion de scoring/ranking y metricas de rendimiento definida.
- [x] Conexion al Hito 3 incluida.
