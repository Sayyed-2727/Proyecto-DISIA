# Capitulo 6. Entrenamiento, evaluación y tablas de resultados (Fase 5)

## 6.1 Descripción de la fase (qué se hace)
En esta fase se ejecuta el entrenamiento final de cada modelo con su mejor configuración de hiperparámetros (identificada en Fase 4), se evalúa en el conjunto de test fijo, y se generan artefactos finales: tablas de métricas, curvas ROC comparadas, calibración y distribución de scores. El resultado es un conjunto de números y visualizaciones que sustentará la discusión científica del Hito 3.

## 6.2 Protocolo operativo detallado

### 6.2.1 Fase de preparación (antes del entrenamiento)
1. Cargar `processed_features_48h_setA.csv` (4000 estancias, ~13.8% prevalencia de mortalidad).
2. Ejecutar split 60/20/20 estratificado con `random_state=42`.
   - Train: 2400 estancias (aprox.)
   - Valid: 800 estancias (aprox.)
   - Test: 800 estancias (aprox.)
3. Verificar distribución de etiqueta en cada split (reportar prevalencia por split).
4. Aplicar pipeline de preprocesado (imputación + escalado según modelo).

### 6.2.2 Entrenamiento por modelo
Para cada modelo en [Logística, Random Forest, XGBoost, (SVM opcional)]:

1. **Búsqueda de hiperparámetros (Fase 4, reiterada aquí con datos finales)**
   - Usar `RandomizedSearchCV` + CV estratificada (5 folds) en conjunto train.
   - Guardar mejores N configuraciones por AUC.

2. **Refinamiento local (opcional)**
   - Si hay tiempo, ejecutar `GridSearchCV` en región prometedora.

3. **Selección de configuración final**
   - Elegir configuración con mayor AUC media en CV, secundariamente por menor Brier.

4. **Reentrenamiento en train+valid**
   - Fusionar train y valid.
   - Reajustar preprocesador (imputador, escalador) en datos fusionados.
   - Reentrenar modelo con configuración final.

5. **Predicción en test**
   - Aplicar pipeline aprendido en train+valid a test.
   - Generaré probabilidades `y_pred_proba` (clase 1).

### 6.2.3 Evaluación única en test
Para cada modelo, calcular:
- **AUC-ROC** (Manning-Whitney, conforme Fase 3).
- **Brier score**.
- **Recall@k** para k ∈ {25, 50, 100, 200}.
- **% fallecidos en top-k** para k ∈ {25, 50, 100, 200}.

Guardar para cada modelo:
- Predicciones en test (`y_test`, `y_pred_proba`).
- Métricas numéricas.
- Artefactos de interpretabilidad (coeficientes en logística, importancias en árboles).

## 6.3 Plantillas de tablas de resultados

Las siguientes tablas ya están **rellenadas con los resultados reales y definitivos** de la corrida consistente `n_iter=30` (seed=42), almacenados en `Hito 3/artifacts/tables/`.

### 6.3.1 Tabla 6.1: Distribución de datos por split

| Split | N estancias | N fallecidos | % prevalencia |
|---|---:|---:|---:|
| Train (60%) | 2400 | 332 | 13.83% |
| Validation (20%) | 800 | 111 | 13.88% |
| Test (20%) | 800 | 111 | 13.88% |
| **Total** | **4000** | **554** | **13.85%** |

> Caption: Distribución de estancias UCI y prevalencia de mortalidad intrahospitalaria por conjunto (Set A, split estratificado 60/20/20, seed=42).

### 6.3.2 Tabla 6.2: Métricas probabilísticas (AUC-ROC y Brier score)

| Modelo | AUC-ROC test | Brier score test | Δ AUC vs baseline Hito 2 | Δ Brier vs baseline Hito 2 (positivo = mejora) |
|---|---:|---:|---:|---:|
| **Baseline Hito 2** | 0.7806 | 0.0887 | — | — |
| Baseline Hito 2 replicado (60/20/20) | 0.8301 | 0.0985 | +0.0495 | -0.0098 |
| Regresión logística | 0.8733 | 0.0879 | +0.0927 | +0.0008 |
| Random Forest | 0.8623 | 0.0902 | +0.0817 | -0.0015 |

> Caption: Desempeño probabilístico de modelos en test fijo (Set A, n=800 estancias). AUC-ROC mide discriminación global; Brier score mide error cuadrático de probabilidades (calibración + discriminación). Los valores se reportan en el conjunto de test independiente y se comparan contra el baseline reproducible de Hito 2 (AUC=0.7806, Brier=0.0887).

### 6.3.3 Tabla 6.3: Métricas de ranking clínico — Recall@k

| Modelo | Recall@25 | Recall@50 | Recall@100 | Recall@200 |
|---|---:|---:|---:|---:|
| **Baseline Hito 2** | 0.1684 | 0.2421 | 0.4526 | 0.6421 |
| Baseline Hito 2 replicado (60/20/20) | 0.1171 | 0.2162 | 0.4234 | 0.6667 |
| Regresión logística | 0.1802 | 0.3063 | 0.4775 | 0.7027 |
| Random Forest | 0.1622 | 0.2883 | 0.4685 | 0.7027 |

> Caption: Proporción de fallecidos capturados en los k primeros pacientes del ranking ordenado por riesgo descendente. Define la utilidad operativa del sistema para triaje clínico: Recall@25 significa "¿cuántos de los 25 pacientes más riesgosos están realmente fallecidos?". Interpretación: valores más altos indican mejor capacidad de concentrar casos de alto riesgo en la zona prioritaria.

### 6.3.4 Tabla 6.4: Métricas de ranking clínico — % fallecidos en top-k

| Modelo | % top-25 | % top-50 | % top-100 | % top-200 |
|---|---:|---:|---:|---:|
| **Baseline Hito 2** | 0.6400 | 0.4600 | 0.4300 | 0.3050 |
| Baseline Hito 2 replicado (60/20/20) | 0.5200 | 0.4800 | 0.4700 | 0.3700 |
| Regresión logística | 0.8000 | 0.6800 | 0.5300 | 0.3900 |
| Random Forest | 0.7200 | 0.6400 | 0.5200 | 0.3900 |

> Caption: Pureza de la zona prioritaria: fracción de fallecidos entre los k pacientes de mayor riesgo. Por ejemplo, si % top-25 = 0.64, significa que de los 25 pacientes más priorizados, el 64% fallecen realmente (vs 13.8% basesline). Esta métrica es clave para operativos de triaje donde el equipo clínico puede atender solo un subconjunto pequeño de forma intensiva.

### 6.3.5 Tabla 6.5: Resumen sintético de mejoras operativas

| Métrica | Baseline Hito 2 (histórico) | Baseline Hito 2 replicado (60/20/20) | Mejor modelo Hito 3 | Ganancia vs histórico | Ganancia vs replicado |
|---|---:|---:|---:|---:|---:|
| AUC-ROC | 0.7806 | 0.8301 | 0.8733 | +0.0927 | +0.0432 |
| Brier (menor es mejor) | 0.0887 | 0.0985 | 0.0879 | +0.0008 | +0.0107 |
| Recall@50 | 0.2421 | 0.2162 | 0.3063 | +0.0642 | +0.0901 |
| % top-50 | 0.4600 | 0.4800 | 0.6800 | +0.2200 | +0.2000 |

> Caption: Síntesis comparativa de mejoras operativas. Permite identificar de un vistazo si el modelo de Hito 3 generaliza mejor (AUC/Brier) y captura más fallecidos en topk de forma más precisa (Recall y pureza).

### 6.3.6 Interpretación breve de resultados finales

En la corrida definitiva (`n_iter=30`), **regresión logística** obtiene el mejor desempeño global en test, con **AUC=0.8733** y **Brier=0.0879**, superando tanto al baseline histórico de Hito 2 (AUC=0.7806, Brier=0.0887) como al baseline replicado en split 60/20/20 (AUC=0.8301, Brier=0.0985). En términos de priorización clínica, la logística también lidera en top-k temprano: `Recall@50=0.3063` y `%top-50=0.6800`, frente a `0.2883` y `0.6400` de Random Forest.

Se observa un **trade-off leve**: Random Forest mantiene rendimiento competitivo en ranking (`Recall@200=0.7027`, igual que logística), pero queda por debajo en calibración/probabilidad (Brier=0.0902). Por tanto, para MIMIC-TRIAGE en esta configuración, logística ofrece el mejor equilibrio entre discriminación, calibración y utilidad operativa top-k.

## 6.4 Propuesta de figuras y visualizaciones

### 6.4.1 Figura 6.1: Curvas ROC comparadas

**Tipo**: Gráfico de curva(s) ROC en mismo eje.

**Descripción**:
- Eje x: Tasa de falsos positivos (1 - Especificidad).
- Eje y: Tasa de verdaderos positivos (Sensibilidad).
- Una curva por modelo (Logística, RF, XGBoost, SVM opcional) + baseline Hito 2 en línea discontínua gris.
- Leyenda con modelo y AUC asociado.

**Elementos clave**:
- Diagonozal (AUC=0.5) como referencia nula.
- Colores distintos para cada modelo (e.g., azul logística, verde RF, rojo XGBoost, naranja SVM).
- Grid suave de fondo.

**Interpretación a incluir en caption**:
- Modelo con curva más arriba y a la izquierda tiene mejor discriminación (AUC más alta).
- Útil para visualizar dominio: ¿algún modelo domina en toda la curva?

### 6.4.2 Figura 6.2: Curvas de calibración (Reliability plots)

**Tipo**: Diagrama de calibración + perfecto (diagonal).

**Descripción**:
- Para cada modelo, dividir predicciones en bins (p. ej., 10 bins por decil de riesgo).
- Eje x: probabilidad predicha media por bin.
- Eje y: frecuencia observada de evento por bin.
- Línea diagonal = calibración perfecta.
- Línea/puntos del modelo = desviación real.

**Elementos clave**:
- Una subgráfica por modelo (o multipanel).
- Diagonal en gris para referencia.
- Puntos con barras de error o áreas de confianza.

**Interpretación a incluir en caption**:
- Modelos arriba de la diagonal: subestiman riesgo (probabilidades bajas en promedio).
- Modelos abajo: sobrestiman riesgo.
- Cercanía a diagonal = buena calibración = confiabilidad probabilística.
- Crítico para triaje clínico: probabilidades mal calibradas pueden distorsionar decisiones.

### 6.4.3 Figura 6.3: Histogramas de scores predichos (fallecidos vs vivos)

**Tipo**: Histogramas superpuestos.

**Descripción**:
- Para cada modelo, plotear distribución de probabilidades predichas.
- Separación por clase: fallecidos (rojo, opaco 0.5) vs no fallecidos (azul, opaco 0.5).
- Eje x: probabilidad predicha [0, 1].
- Eje y: frecuencia o densidad.

**Elementos clave**:
- Histogramas con binning adecuado (p. ej., 30-40 bins).
- Etiqueta clara de clase y N de eventos.
- Título con nombre del modelo.

**Interpretación a incluir en caption**:
- Separación clara entre rojo (fallecidos) y azul (vivos) indica buena discriminación.
- Solapamiento importante sugiere dificultades en casos límite.
- Forma de distribución (bimodal vs unimodal) informa sobre estructura del modelo.

## 6.5 Protocolo de generación de artefactos

1. **Tablas en formato Markdown**
   - Editables facilmente para rellenar con resultados.
   - Incluyendo columnas de comentarios o notas si es necesario.

2. **Figuras**
   - Generar con matplotlib/seaborn (preferible) o base R si es caso.
   - Guardar en formato PNG/PDF a 300 dpi para calidad de paper.
   - Leyendas y captions auto-generados con valores reales.

3. **Reproducibilidad**
   - Guardar semillas y configuraciones en cada experimento.
   - Conservar logs de CV para auditar selección de hiperparámetros.
   - Posibilidad de recalcular tablas/figuras a partir de artefactos (predictions, métricas).

## 6.6 Checklist Fase 5 (operativo)
- [x] Descripción completa del protocolo train/valid/test.
- [x] Plantillas de tablas (al menos una por tipo de métrica).
- [x] Propuesta de figuras con interpretación operativa.
- [x] Especificación clara de qué números van en cada slot.

## 6.7 Entrada esperada de la Fase 5 (después de ejecutar experimentos)

Tras completar la ejecución computacional:
- Resultados parciales de búsqueda de hiperparámetros (AUC/Brier en CV).
- Mejores configuraciones por modelo.
- Predicciones en test y métricas finales (para rellenar tablas).
- Gráficas generadas (ROC, calibración, histogramas).

Estos artefactos se usarán en Fase 6 (interpretabilidad) y Fase 7 (redacción final).

## 6.8 Notas metodológicas para evitar sesgos comunes

1. **Evaluación única en test**: No tocar test hasta que la configuración final esté cerrada. Sin peeking.
2. **Comparación justa**: Todos los modelos evaluados en el *mismo* conjunto de test.
3. **Reportar incertidumbre**: Si es posible, incluir intervalos de confianza en bootstrap (p. ej., AUC ± IC95%).
4. **Documentar decisiones**: Cada paso de selección debe quedar registrado para auditoría.
5. **Coherencia de prevalencia**: Verificar que test tiene ~13.8% prevalencia (consistencia con Set A global).
