# Hito 3 - Modelado, experimentación y validación
## Proyecto MIMIC-TRIAGE (adaptado a PhysioNet/CinC 2012)

## 1. Objetivo del Hito 3

El objetivo de este hito es desarrollar la fase de modelado del proyecto, diseñando y validando experimentalmente distintos modelos de aprendizaje automático para estimar el riesgo de mortalidad intrahospitalaria y generar un ranking clínico de priorización.

A diferencia del Hito 2, centrado en el análisis exploratorio de los datos y en la formulación del problema, en este hito el foco se traslada a la comparación de modelos, la evaluación rigurosa de su rendimiento y la selección de una solución principal que pueda considerarse adecuada para el caso de uso.

En concreto, este hito persigue tres metas principales:

1. definir un baseline reproducible;
2. comparar distintos modelos candidatos bajo una metodología homogénea;
3. validar los resultados desde dos perspectivas complementarias: calidad probabilística y utilidad operativa para ranking clínico.

---

## 2. Punto de partida y datos utilizados

El modelado parte del dataset tabular `processed_features_48h_setA.csv`, construido a partir de los datos raw de PhysioNet/CinC 2012 correspondientes a Set A.

Aunque el origen de los datos es temporal, en este trabajo no se realiza modelado secuencial explícito. Las mediciones registradas durante las primeras 48 horas de cada estancia se transforman en una representación tabular agregada por paciente, utilizando estadísticas resumen como valor inicial, valor final, media, mínimo, máximo, desviación típica, número de mediciones y bandera de presencia.

Esta decisión metodológica permite trabajar con modelos tabulares supervisados y facilita la comparación experimental, la validación y la interpretabilidad, manteniendo al mismo tiempo información clínica relevante derivada de la evolución temprana del paciente.

El dataset final contiene una fila por estancia y una variable objetivo binaria, `In-hospital_death`, que indica si el paciente falleció durante el ingreso hospitalario.

---

## 3. Formulación del problema

El problema se plantea como una tarea de **scoring probabilístico de riesgo**, donde cada estancia recibe una probabilidad estimada de mortalidad intrahospitalaria.

A partir de ese score continuo, las estancias pueden ordenarse de mayor a menor riesgo para construir un **ranking clínico de priorización**. Esta formulación permite evaluar el sistema no solo como clasificador binario, sino también como herramienta de apoyo a la toma de decisiones en escenarios de triaje o asignación limitada de recursos.

Por tanto, el rendimiento se analiza desde dos perspectivas:

- **métricas probabilísticas**, para medir discriminación y calidad del score;
- **métricas top-k**, para medir la capacidad del modelo de concentrar pacientes fallecidos en la parte alta del ranking.

---

## 4. Variables empleadas en el baseline

Como punto de partida se definió un baseline basado en regresión logística sobre un subconjunto reducido de variables clínicas y demográficas seleccionadas manualmente.

Las variables utilizadas fueron:

- `Age_first`
- `Gender_first`
- `ICUType_first`
- `GCS_mean`
- `HR_mean`
- `MAP_mean`
- `SysABP_mean`
- `DiasABP_mean`
- `RespRate_mean`
- `Temp_mean`
- `Creatinine_mean`
- `BUN_mean`
- `WBC_mean`
- `Urine_mean`

La selección de este conjunto responde a dos criterios principales: simplicidad e interpretabilidad. Se priorizó un subconjunto compacto pero clínicamente razonable, suficiente para establecer una referencia inicial antes de explorar modelos más complejos.

---

## 5. Baseline experimental

### 5.1. Justificación del modelo baseline

El primer experimento se construyó con regresión logística. Este modelo se seleccionó como baseline por varias razones:

1. es un modelo estándar y ampliamente utilizado en problemas de predicción de riesgo clínico;
2. proporciona una referencia interpretable;
3. permite estimar directamente probabilidades continuas de mortalidad;
4. resulta adecuado como punto de comparación para modelos no lineales posteriores.

### 5.2. Implementación

Inicialmente se desarrolló una implementación manual del baseline con imputación por mediana, estandarización y ajuste de regresión logística mediante descenso del gradiente. Posteriormente, este baseline se reimplementó con `scikit-learn` para disponer de una versión más estándar, reproducible y fácil de extender a la comparación de modelos.

La versión oficial del baseline quedó definida mediante:

- imputación por mediana;
- estandarización;
- regresión logística con `scikit-learn`.

La partición inicial entrenamiento/test se construyó de forma determinista a partir de `RecordID`, con 3195 estancias para entrenamiento y 805 para test.

### 5.3. Resultados del baseline

El baseline obtuvo en test:

- **AUC-ROC:** 0.7806
- **Brier score:** 0.0887

Además, desde el punto de vista del ranking clínico:

- **Recall@25:** 0.1684
- **Recall@50:** 0.2421
- **Recall@100:** 0.4526
- **Recall@200:** 0.6421

La prevalencia de mortalidad en test fue aproximadamente del 11.8%, mientras que el porcentaje de fallecidos en el top-25 alcanzó el 64%, lo que confirma que incluso este baseline sencillo ya concentra riesgo real en la parte alta del ranking.

En conjunto, estos resultados muestran que el dataset procesado contiene señal suficiente para abordar el problema como una tarea de scoring probabilístico y priorización clínica.

---

## 6. Comparación inicial de modelos

Una vez definido el baseline, se incorporaron dos modelos no lineales para comparación inicial:

- **Random Forest**
- **HistGradientBoostingClassifier**

Ambos modelos se evaluaron usando exactamente:

- el mismo subconjunto de variables;
- la misma partición entrenamiento/test;
- las mismas métricas probabilísticas;
- y las mismas métricas top-k de ranking clínico.

### 6.1. Random Forest

Random Forest se seleccionó por ser un modelo especialmente adecuado para datos tabulares, capaz de capturar relaciones no lineales e interacciones entre variables sin requerir un preprocesado complejo.

Resultados en test:

- **AUC-ROC:** 0.8520
- **Brier score:** 0.0818

Top-k:

- **Recall@25:** 0.1789
- **Recall@50:** 0.2947
- **Recall@100:** 0.4632
- **% fallecidos top-25:** 0.68

Estos resultados mejoran claramente al baseline lineal.

### 6.2. Gradient Boosting

Gradient Boosting se incorporó como segundo modelo basado en árboles, con el objetivo de comprobar si una estrategia de boosting podía superar a Random Forest.

Resultados en test:

- **AUC-ROC:** 0.8401
- **Brier score:** 0.0839

Top-k:

- **Recall@25:** 0.1579
- **Recall@50:** 0.2947
- **Recall@100:** 0.4632
- **% fallecidos top-25:** 0.60

Aunque mejora respecto a la regresión logística, no supera a Random Forest en esta comparación inicial.

### 6.3. Interpretación de la comparación inicial

La comparación inicial mostró que los modelos basados en árboles mejoran de forma clara al baseline lineal. Esto sugiere que el problema contiene relaciones no lineales e interacciones entre variables clínicas que la regresión logística no captura completamente.

En esta primera fase, Random Forest emergió como el modelo más fuerte, con mejor AUC-ROC, mejor Brier score y mejor comportamiento top-k en la partición test utilizada.

---

## 7. Validación cruzada estratificada

Dado que una única partición train/test no es suficiente para extraer conclusiones sólidas, se realizó una validación cruzada estratificada de 5 folds sobre los tres modelos candidatos:

- Logistic Regression
- Random Forest
- Gradient Boosting

Las métricas consideradas en esta fase fueron:

- **AUC-ROC**
- **Brier score**

### 7.1. Resultados medios

Los resultados medios en validación cruzada fueron:

- **Gradient Boosting**
  - AUC-ROC medio: 0.8254
  - Brier medio: 0.0969

- **Random Forest**
  - AUC-ROC medio: 0.8222
  - Brier medio: 0.0963

- **Logistic Regression**
  - AUC-ROC medio: 0.7866
  - Brier medio: 0.1026

### 7.2. Interpretación

La validación cruzada confirmó que los modelos basados en árboles superan consistentemente al baseline lineal. Sin embargo, también mostró que la diferencia entre Random Forest y Gradient Boosting es mucho más ajustada de lo que parecía en la partición única inicial.

En promedio:

- Gradient Boosting presentó el mejor AUC medio;
- Random Forest presentó el mejor Brier medio.

Como las diferencias fueron pequeñas y la dispersión entre folds similar, ambos modelos se consideraron candidatos competitivos.

---

## 8. Validación cruzada orientada a ranking clínico

Dado que el objetivo operativo del sistema es priorizar pacientes, se amplió la validación cruzada incorporando métricas top-k sobre los dos modelos principales:

- Random Forest
- Gradient Boosting

Se analizaron:

- **Recall@25**
- **Recall@50**
- **Recall@100**
- porcentaje de fallecidos en top-k

### 8.1. Resultados

Los resultados medios mostraron que:

- **Gradient Boosting** obtuvo una ligera ventaja en el **top-25**;
- **Random Forest** mostró mejores resultados medios en **top-50** y **top-100**.

### 8.2. Interpretación

La comparación top-k reveló que no existe un dominador absoluto en todos los cortes del ranking. Gradient Boosting se comporta ligeramente mejor en la zona más extrema del ranking, mientras que Random Forest ofrece un comportamiento más equilibrado al ampliar el número de pacientes priorizados.

En conjunto, ambos modelos resultan adecuados para el caso de uso, aunque Random Forest fue considerado el candidato más equilibrado al combinar validación probabilística y validación orientada a ranking.

---

## 9. Ajuste de hiperparámetros del modelo seleccionado

Una vez identificado Random Forest como candidato principal, se realizó un ajuste acotado de hiperparámetros mediante `GridSearchCV` y validación cruzada estratificada.

La búsqueda incluyó:

- número de árboles;
- profundidad máxima;
- tamaño mínimo de hoja;
- número de variables consideradas en cada partición.

### 9.1. Mejor configuración encontrada

La mejor combinación obtenida fue:

- `n_estimators = 500`
- `max_depth = None`
- `min_samples_leaf = 5`
- `max_features = 'sqrt'`

con un **AUC medio en validación cruzada de 0.8253**.

### 9.2. Interpretación del tuning

La mejora respecto a la configuración inicial fue moderada, lo que indica que el modelo ya partía de una parametrización razonable. Además, las mejores configuraciones presentaron un patrón consistente, lo que sugiere que el rendimiento del modelo es relativamente estable y no depende de un ajuste extremadamente fino.

---

## 10. Evaluación final del Random Forest ajustado

El Random Forest ajustado se volvió a evaluar sobre el conjunto de test original.

### 10.1. Resultados

Resultados probabilísticos:

- **AUC-ROC:** 0.8554
- **Brier score:** 0.0815

Resultados top-k:

- **Recall@25:** 0.1684
- **Recall@50:** 0.3263
- **Recall@100:** 0.4632
- **% fallecidos top-25:** 0.64

### 10.2. Interpretación

El ajuste produjo una mejora ligera en las métricas probabilísticas y en algunos cortes del ranking, especialmente en Recall@50. Sin embargo, no mejoró todos los tramos del ranking de manera uniforme, ya que el modelo inicial mantenía una ligera ventaja en top-25.

Aun así, considerando el conjunto de evidencias, el Random Forest ajustado se seleccionó como modelo principal del Hito 3 por ofrecer el mejor equilibrio global entre discriminación, calidad probabilística y utilidad operativa.

---

## 11. Interpretabilidad del modelo final

### 11.1. Importancia global de variables

Como primera aproximación a la interpretabilidad, se analizó la importancia global de variables del Random Forest ajustado.

Las variables más relevantes fueron:

- `GCS_mean`
- `Urine_mean`
- `BUN_mean`
- `WBC_mean`
- `Temp_mean`
- `HR_mean`
- `Creatinine_mean`
- `Age_first`
- `SysABP_mean`

Este patrón es clínicamente plausible, ya que combina información sobre:

- estado neurológico;
- función renal y balance hídrico;
- respuesta inflamatoria/infecciosa;
- estabilidad hemodinámica;
- y riesgo basal asociado a la edad.

### 11.2. Interpretabilidad con SHAP

Como complemento, se realizó un análisis global basado en SHAP sobre una muestra del conjunto de test.

Las variables con mayor impacto medio absoluto fueron:

- `GCS_mean`
- `BUN_mean`
- `Urine_mean`
- `Age_first`
- `Temp_mean`

SHAP confirmó en gran medida el patrón observado en la importancia global del Random Forest. La coincidencia entre ambos enfoques refuerza la idea de que el modelo se apoya en señales clínicamente razonables y no en patrones arbitrarios.

Además, variables como `Gender_first` mostraron una contribución muy reducida, lo que sugiere que el modelo no depende fuertemente de ellas.

---

## 12. Modelo final seleccionado

A partir del conjunto de experimentos realizados, se establece la siguiente jerarquía final:

- **Baseline interpretable:** Logistic Regression
- **Competidor cercano:** Gradient Boosting
- **Modelo principal seleccionado:** Random Forest ajustado

La elección final del Random Forest ajustado se justifica por su buen comportamiento global, su solidez en validación cruzada, su rendimiento competitivo en métricas top-k y su interpretabilidad razonable.

---

## 13. Conclusiones

En este Hito 3 se ha completado la fase de modelado, experimentación y validación del proyecto MIMIC-TRIAGE sobre la adaptación a PhysioNet/CinC 2012.

A partir del dataset tabular agregado por estancia, se construyó un baseline interpretable con regresión logística y se comparó con dos modelos no lineales basados en árboles: Random Forest y Gradient Boosting. Los resultados mostraron de forma consistente que los modelos basados en árboles superan al baseline lineal, lo que indica que el problema presenta relaciones no lineales relevantes entre variables clínicas.

La validación cruzada probabilística y la validación cruzada orientada a ranking clínico permitieron concluir que Random Forest y Gradient Boosting son ambos modelos competitivos, aunque Random Forest ofrece el mejor equilibrio global en esta fase. Tras un ajuste acotado de hiperparámetros, el Random Forest ajustado se seleccionó como modelo principal del Hito 3.

Finalmente, el análisis de interpretabilidad mediante importancia global de variables y SHAP mostró que el modelo se apoya en señales clínicamente plausibles, especialmente relacionadas con estado neurológico, función renal, balance hídrico, respuesta inflamatoria y estabilidad hemodinámica.

En conjunto, este hito deja validada una solución de modelado adecuada para el problema planteado y proporciona una base sólida para la siguiente fase de documentación e integración del sistema.