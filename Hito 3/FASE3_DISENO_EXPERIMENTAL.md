# Capitulo 4. Diseño experimental (Fase 3)

## 4.1 Descripción de la fase (qué se hace)
En esta fase se define el protocolo experimental completo para comparar modelos de MIMIC-TRIAGE de forma rigurosa y reproducible. El diseño fija: (i) cómo se particionan los datos, (ii) cómo se construye el pipeline de preprocesado, (iii) cómo se trata el desbalanceo de clases y (iv) qué métricas se optimizan y reportan.

Se mantiene coherencia con el alcance del proyecto: predicción de mortalidad intrahospitalaria en ventana 0-48h, con salida en forma de scoring probabilístico y ranking clínico. También se conserva coherencia con la documentación previa: el Challenge 2012 contiene aproximadamente 12.000 estancias UCI adultas (Set A/B/C), mientras que en Hito 2 se trabajó específicamente con Set A etiquetado (4000 estancias) para construir el baseline y los artefactos de features.

## 4.2 Partición de datos y validación

### 4.2.1 Cohorte de trabajo para Hito 3
- **Dataset de modelado principal**: `processed_features_48h_setA.csv` (4000 estancias etiquetadas).
- **Unidad de análisis**: una fila por estancia (`RecordID`).
- **Variable objetivo**: `In-hospital_death` (0/1).

### 4.2.2 Esquema de partición propuesto
Se define un esquema **estratificado 60/20/20** por etiqueta:
- **Train**: 60%
- **Validación**: 20%
- **Test**: 20%

Configuración reproducible:
- `random_state = 42`
- estratificación por `In-hospital_death`
- comparación de todos los modelos en el **mismo conjunto de test**

Implementación recomendada:
1. `train_test_split` estratificado para separar `test` (20%).
2. Segundo `train_test_split` estratificado sobre el bloque restante para obtener `train` (60%) y `valid` (20%).

> Nota de consistencia con Hito 2: el baseline previo se evaluó con split reproducible por `RecordID % 5 == 0`. En Hito 3 se adopta 60/20/20 estratificado para tuning más estable, manteniendo comparabilidad mediante evaluación final en un test fijo y reportando claramente ambos esquemas cuando se compare contra el baseline histórico.

### 4.2.3 Validación para selección de hiperparámetros
Sobre el conjunto de entrenamiento (60%) se aplicará validación cruzada estratificada:
- **StratifiedKFold** con `n_splits=5`, `shuffle=True`, `random_state=42`
- La selección de hiperparámetros se hará maximizando AUC y controlando Brier en validación.

## 4.3 Pipeline de preprocesado

## 4.3.1 Entrada del modelo
Vector por estancia con:
- **Variables estáticas** (edad, sexo, peso, altura, tipo de UCI).
- **Variables temporales agregadas 0-48h** (e.g., `first`, `last`, `mean`, `min`, `max`, `std`, `n_mediciones`, `flag_medido`).

### 4.3.2 Limpieza y control de tipos
- Conversión explícita a numérico de columnas continuas.
- Verificación de duplicados por `RecordID`.
- Exclusión de variables con leakage temporal o de outcome.
- Registro de columnas finales usadas por cada experimento.

### 4.3.3 Tratamiento de faltantes
Estrategia base:
- **Imputación por mediana** para variables continuas.
- Conservación de señales de missingness informativo con `flag_medido` y/o `n_mediciones`.

Regla de implementación:
- El imputador se **ajusta solo en train** y se aplica a valid/test sin recalcular estadísticas.

### 4.3.4 Escalado de variables
- Para modelos sensibles a escala (Regresión Logística, SVM, MLP): **estandarización z-score**.
- Para modelos de árboles (RF, GBM/XGBoost): escalado opcional/no requerido.

Regla de implementación:
- El escalador se **ajusta solo en train** y se reutiliza en valid/test.

### 4.3.5 Codificación de variables categóricas
- `Gender` e `ICUType` se mantienen en formato numérico consistente con Hito 2.
- Si se requiere, se evaluará one-hot en modelos lineales para contraste metodológico, dejando trazabilidad del cambio.

### 4.3.6 Encapsulado del pipeline
Uso recomendado de `Pipeline` / `ColumnTransformer` de scikit-learn para:
- evitar fuga de información,
- garantizar reproducibilidad,
- facilitar comparación homogénea entre modelos.

## 4.4 Tratamiento del desbalanceo de clases

En Set A, la clase positiva (mortalidad) es minoritaria (~13.8%), por lo que no es suficiente optimizar solo exactitud global.

Estrategia propuesta (orden de prioridad):
1. **`class_weight='balanced'`** en modelos que lo soporten (Logistic Regression, SVM, Random Forest).
2. Ajuste de umbral operativo solo para análisis secundarios (el ranking principal usará probabilidad continua).
3. Sobremuestreo/submuestreo (SMOTE o similares) solo como experimento adicional, no como configuración principal, para evitar introducir sesgos por síntesis en variables clínicas.

Criterio de decisión:
- Se priorizarán configuraciones que mejoren AUC y Recall@k sin degradar de forma importante Brier (calibración).

## 4.5 Definición matemática de métricas

Sea un conjunto de test con $N$ estancias, etiqueta real $y_i \in \{0,1\}$ y probabilidad predicha $\hat{p}_i$.

### 4.5.1 AUC-ROC
La AUC puede interpretarse como la probabilidad de que un positivo tenga score mayor que un negativo:

$$
\mathrm{AUC} = P(\hat{p}^+ > \hat{p}^-)
$$

Equivalentemente, vía ranking de Mann-Whitney:

$$
\mathrm{AUC} = \frac{\sum_{i:y_i=1} \mathrm{rank}(\hat{p}_i) - \frac{n_+(n_+ + 1)}{2}}{n_+ n_-}
$$

donde $n_+$ y $n_-$ son el número de positivos y negativos.

### 4.5.2 Brier score
Error cuadrático medio de probabilidades:

$$
\mathrm{Brier} = \frac{1}{N}\sum_{i=1}^{N}(\hat{p}_i - y_i)^2
$$

Menor valor implica mejor calidad probabilística global (discriminación + calibración).

### 4.5.3 Recall@k
Se ordenan pacientes por riesgo descendente y se toma el conjunto top-$k$:

$$
\mathrm{Recall@}k = \frac{\sum_{i \in \mathrm{Top}k} y_i}{\sum_{i=1}^{N} y_i}
$$

Mide la fracción de fallecidos capturados dentro de los primeros $k$ pacientes priorizados.

### 4.5.4 % fallecidos en top-k
Pureza de la zona prioritaria del ranking:

$$
\%\mathrm{fallecidos\ en\ top\text{-}k} = \frac{\sum_{i \in \mathrm{Top}k} y_i}{k}
$$

Se reportará para $k \in \{25, 50, 100, 200\}$ para mantener continuidad con Hito 2.

## 4.6 Protocolo operativo de evaluación

1. Fijar partición 60/20/20 estratificada (`seed=42`).
2. Definir pipeline de preprocesado por familia de modelo.
3. Seleccionar hiperparámetros con CV estratificada en train.
4. Reentrenar configuración final en train+valid.
5. Evaluar una única vez en test fijo con AUC, Brier, Recall@k y % top-k.
6. Guardar artefactos: predicciones, tabla de métricas, configuración y semilla.

## 4.7 Resultados esperables de la fase

### 4.7.1 Texto
- Sección metodológica lista para paper con justificación técnica de partición, preprocesado y control de sesgos.

### 4.7.2 Tablas
- Tabla de partición y prevalencia por split (train/valid/test).
- Tabla-resumen del pipeline por familia de modelos (lineales vs árboles).
- Tabla de definición formal de métricas usadas en evaluación final.

### 4.7.3 Pseudocódigo sugerido
```text
Input: processed_features_48h_setA.csv
Split estratificado 60/20/20 (seed=42)
For each modelo candidato:
    Construir pipeline (imputación + escalado si aplica + clasificador)
    Ajustar hiperparámetros con CV estratificada (train)
Seleccionar mejor configuración por AUC/Brier (validación)
Reentrenar en train+valid
Evaluar en test fijo -> AUC, Brier, Recall@k, % top-k
Guardar métricas y predicciones para tablas/figuras
```

## 4.8 Checklist Fase 3
- [x] Esquema de partición de datos definido y justificado.
- [x] Pipeline de preprocesado descrito paso a paso.
- [x] Estrategia para tratar el desbalanceo de clases.
- [x] Definición matemática de las métricas usada en el Hito 3.
