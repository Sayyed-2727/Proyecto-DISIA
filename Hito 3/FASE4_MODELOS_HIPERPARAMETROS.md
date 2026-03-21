# Capitulo 5. Selección de modelos y búsqueda de hiperparámetros (Fase 4)

## 5.1 Descripción de la fase (qué se hace)
En esta fase se define el conjunto de modelos que se compararán en MIMIC-TRIAGE para predicción de mortalidad intrahospitalaria en ventana 0-48h, junto con su estrategia de ajuste de hiperparámetros. El objetivo es equilibrar tres criterios: (i) rendimiento predictivo, (ii) robustez ante desbalanceo y missingness, y (iii) interpretabilidad clínica.

La comparación se diseña en continuidad con Hito 2 (baseline lineal) y con el estado del arte del Challenge 2012 y literatura posterior: se incluirá un modelo lineal interpretable, modelos de ensamble basados en árboles y un modelo adicional opcional para contraste de frontera no lineal.

## 5.2 Modelos seleccionados y justificación

### 5.2.1 Regresión logística (baseline interpretativo)
**Rol en el estudio**
- Modelo de referencia principal por su interpretabilidad y trazabilidad clínica.
- Permite comparar directamente con el baseline de Hito 2.

**Ventajas**
- Coeficientes interpretables (dirección y magnitud del efecto).
- Buena estabilidad con pipeline tabular y regularización.
- Entrenamiento rápido y reproducible.

**Limitaciones**
- Capacidad limitada para capturar interacciones no lineales complejas.
- Sensible a escala y colinealidad si no se preprocesa adecuadamente.

### 5.2.2 Random Forest
**Rol en el estudio**
- Primer modelo no lineal robusto para capturar interacciones entre variables agregadas 0-48h.

**Ventajas**
- Menor sensibilidad a escala.
- Robusto a ruido y sobreajuste moderado por promediado de árboles.
- Permite importancia de variables de forma directa.

**Limitaciones**
- Probabilidades a veces poco calibradas sin post-procesado.
- Menor interpretabilidad local que un modelo lineal.

### 5.2.3 Gradient Boosting / XGBoost
**Rol en el estudio**
- Modelo de alta capacidad para mejorar discriminación y ranking clínico top-k.

**Ventajas**
- Excelente rendimiento en datos tabulares heterogéneos.
- Captura no linealidades e interacciones de orden alto.
- Flexible para manejar desbalanceo (`scale_pos_weight` en XGBoost).

**Limitaciones**
- Mayor sensibilidad a hiperparámetros.
- Riesgo de sobreajuste si no se controla profundidad, learning rate y regularización.

### 5.2.4 SVM (opcional recomendado)
**Rol en el estudio**
- Contraste metodológico con frontera no lineal (kernel RBF) sobre un subconjunto de configuraciones.

**Ventajas**
- Puede rendir bien en problemas de clasificación desbalanceados con tuning adecuado.

**Limitaciones**
- Coste computacional mayor en datasets medianos/grandes.
- Menor escalabilidad y calibración probabilística indirecta (si se activa `probability=True`).

> Decisión práctica: SVM se ejecutará como bloque opcional. Si coste/tiempo es alto, se prioriza logística + RF + XGBoost como núcleo del Hito 3.

## 5.3 Espacios de hiperparámetros

## 5.3.1 Regresión logística
Modelo: `sklearn.linear_model.LogisticRegression`

Parámetros a explorar:
- `penalty`: [`l2`, `l1`] (si solver compatible)
- `C`: [0.01, 0.1, 1, 5, 10, 50]
- `solver`: [`liblinear`, `saga`]
- `class_weight`: [None, `balanced`]
- `max_iter`: [1000, 3000]

Notas:
- Requiere estandarización de variables numéricas.
- Se seleccionará configuración con mejor trade-off AUC/Brier en validación.

## 5.3.2 Random Forest
Modelo: `sklearn.ensemble.RandomForestClassifier`

Parámetros a explorar:
- `n_estimators`: [200, 400, 800]
- `max_depth`: [None, 6, 10, 16]
- `min_samples_split`: [2, 10, 25]
- `min_samples_leaf`: [1, 5, 10]
- `max_features`: [`sqrt`, `log2`, 0.5]
- `class_weight`: [None, `balanced`, `balanced_subsample`]

Notas:
- No requiere escalado estricto.
- Se analizará estabilidad de importancias entre folds.

## 5.3.3 Gradient Boosting / XGBoost
Modelo recomendado: `xgboost.XGBClassifier` (si librería disponible).  
Alternativa si no está disponible: `sklearn.ensemble.GradientBoostingClassifier`.

Parámetros a explorar (XGBoost):
- `n_estimators`: [200, 400, 800]
- `learning_rate`: [0.01, 0.03, 0.05, 0.1]
- `max_depth`: [3, 4, 6, 8]
- `min_child_weight`: [1, 3, 5]
- `subsample`: [0.7, 0.85, 1.0]
- `colsample_bytree`: [0.6, 0.8, 1.0]
- `reg_alpha`: [0, 0.1, 1]
- `reg_lambda`: [1, 5, 10]
- `scale_pos_weight`: [1, 3, 5, 7] (ajustable según prevalencia)

Notas:
- Se recomienda `early_stopping_rounds` usando validación interna para controlar sobreajuste.
- Puede requerir calibración posterior de probabilidades (fase de resultados).

## 5.3.4 SVM (opcional)
Modelo: `sklearn.svm.SVC`

Parámetros a explorar:
- `kernel`: [`rbf`]
- `C`: [0.5, 1, 2, 5, 10]
- `gamma`: [`scale`, 0.1, 0.01, 0.001]
- `class_weight`: [None, `balanced`]
- `probability`: [True]

Notas:
- Requiere estandarización.
- Se limitará el número de combinaciones para controlar coste computacional.

## 5.4 Estrategia de búsqueda de hiperparámetros

### 5.4.1 Enfoque general
Se utilizará una estrategia por etapas, priorizando eficiencia y comparabilidad:

1. **Búsqueda inicial amplia**
   - `RandomizedSearchCV` (30-60 combinaciones por modelo según complejidad).
   - Validación cruzada estratificada de 5 folds (`StratifiedKFold`, `shuffle=True`, `random_state=42`).

2. **Refinamiento local**
   - `GridSearchCV` reducido alrededor de los mejores hiperparámetros de la etapa 1.

3. **Selección final**
   - Modelo ganador por desempeño en validación (AUC principal, Brier como criterio de calibración).
   - Reentrenamiento en `train + valid` y evaluación única en `test` fijo.

### 5.4.2 Criterios de scoring durante la búsqueda
Scoring en CV:
- Primario: `roc_auc`
- Secundario: `neg_brier_score`

Criterio de desempate recomendado:
1. Mayor AUC media en CV.
2. Menor Brier medio en CV.
3. Menor desviación estándar entre folds (robustez).

### 5.4.3 Control de reproducibilidad y trazabilidad
- Fijar `random_state=42` en todos los modelos que lo soporten.
- Guardar por experimento: hiperparámetros, métricas CV, métrica en validación y métrica final en test.
- Mantener mismo conjunto de test para todos los modelos.

## 5.5 Configuración recomendada (presupuesto académico)
Para equilibrio entre calidad y tiempo de ejecución:
- Logística: ~20-30 combinaciones.
- Random Forest: ~30-50 combinaciones.
- XGBoost: ~40-80 combinaciones.
- SVM opcional: ~15-25 combinaciones.

Con este presupuesto, el estudio es suficientemente amplio para justificar resultados en un informe académico sin sobredimensionar el coste computacional.

## 5.6 Resultados esperables de la fase

### 5.6.1 Texto
- Sección metodológica que justifique por qué esos modelos son apropiados para MIMIC-TRIAGE.
- Explicación transparente de cómo se eligieron hiperparámetros y cómo se evitó sobreajuste.

### 5.6.2 Tablas sugeridas
1. **Tabla de modelos candidatos** (tipo, ventajas, limitaciones, rol en el estudio).
2. **Tabla de hiperparámetros por modelo** (espacio de búsqueda).
3. **Tabla de configuración final seleccionada** (a completar tras Fase 5).

### 5.6.3 Pseudocódigo sugerido
```text
For model in [Logistic, RF, XGBoost, (SVM opcional)]:
    Definir pipeline de preprocesado específico
    Ejecutar RandomizedSearchCV (CV estratificada)
    Refinar con GridSearchCV local
    Guardar mejor configuración por AUC/Brier
Seleccionar configuración final por modelo
Reentrenar en train+valid
Reservar test para evaluación final (Fase 5)
```

## 5.7 Checklist Fase 4
- [x] Lista de modelos seleccionados con justificación.
- [x] Hiperparámetros clave y rangos definidos.
- [x] Estrategia de búsqueda de hiperparámetros explicada.
