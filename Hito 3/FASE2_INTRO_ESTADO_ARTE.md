# Capitulo 3. Introduccion y estado del arte

## 3.1 Introduccion

Las unidades de cuidados intensivos (UCI) concentran pacientes con alta inestabilidad hemodinamica, respiratoria y metabolica, en contextos donde las decisiones de priorizacion deben tomarse en ventanas temporales cortas y con informacion incompleta. En este escenario, la mortalidad intrahospitalaria es un desenlace clinicamente relevante para caracterizar gravedad y para evaluar sistemas de soporte a la decision. Sin embargo, en practica asistencial, la evaluacion del riesgo suele combinar experiencia clinica, reglas de puntuacion y observacion continua, lo que introduce variabilidad entre profesionales, turnos y servicios.

MIMIC-TRIAGE se plantea como un sistema de priorizacion temprana que estima el riesgo de mortalidad intrahospitalaria a partir de datos disponibles en las primeras 0-48 horas de estancia UCI. El objetivo operativo no se limita a una etiqueta binaria, sino que produce un score probabilistico por paciente y, a partir de ese score, un ranking de riesgo para triaje. Esta formulacion tiene dos ventajas: (i) preserva la interpretabilidad probabilistica para analisis de calibracion y (ii) traduce la prediccion a una herramienta accionable para priorizacion de recursos, seguimiento y vigilancia clinica.

El marco de datos utilizado procede del PhysioNet/Computing in Cardiology Challenge 2012 (v1.0.0), basado en cohortes de MIMIC-II, con alrededor de 12.000 estancias UCI adultas distribuidas en subconjuntos y con outcomes de mortalidad intrahospitalaria. Cada estancia combina descriptores estaticos (edad, sexo, peso, altura, tipo de UCI) y series temporales de constantes y analiticas durante las primeras 48 horas. Sobre este material, en Hito 2 se definio una representacion tabular por estancia mediante agregaciones temporales (medias, minimos, maximos, dispersion, volumen de medicion y flags de ausencia), junto con un baseline probabilistico reproducible.

Este Hito 3 desarrolla la fase cientifica del proyecto: comparar modelos de machine learning sobre el mismo pipeline de features 0-48h, justificar su idoneidad metodologica, evaluar su rendimiento probabilistico y de ranking clinico, y analizar su interpretabilidad. El enfoque busca equilibrio entre validez tecnica (AUC, Brier, robustez) y utilidad operativa (Recall@k y concentracion de fallecidos en top-k), alineando el diseño experimental con el uso real de una lista de priorizacion clinica.

## 3.2 Estado del arte

### 3.2.1 Scores de gravedad tradicionales en UCI: APACHE II, SAPS y SOFA

Antes de la adopcion masiva de modelos de machine learning, la estratificacion de riesgo en UCI se ha apoyado en escalas de severidad como APACHE II, SAPS y SOFA. Estas herramientas han sido fundamentales para estandarizar la evaluacion de gravedad, comparar poblaciones entre centros y estimar pronostico agregado. En particular, han contribuido a introducir una cultura de decision basada en evidencia cuantitativa, no solo en juicio experto.

No obstante, estos scores presentan limitaciones conocidas cuando se buscan decisiones de priorizacion fina a nivel de paciente individual. Suelen apoyarse en un subconjunto acotado de variables y reglas fijas de puntuacion, con capacidad limitada para capturar interacciones no lineales complejas, dinamica temporal irregular o patrones de missingness informativo. Ademas, su rendimiento puede variar entre cohortes y periodos asistenciales, y no siempre ofrecen una calibracion consistente fuera del contexto donde fueron construidos.

Desde la perspectiva de MIMIC-TRIAGE, estos scores se consideran un marco de referencia clinica y metodologica, pero no un techo de rendimiento. El objetivo es conservar su valor interpretativo y de uso practico, incorporando al mismo tiempo la flexibilidad de modelos de aprendizaje estadistico sobre representaciones mas ricas de la ventana 0-48h.

### 3.2.2 PhysioNet/Computing in Cardiology Challenge 2012 como benchmark

El Challenge 2012 establecio un benchmark influyente para prediccion de mortalidad intrahospitalaria temprana en UCI. La tarea se organizo en dos eventos complementarios: un evento de clasificacion (discriminar fallece/no fallece) y un evento de estimacion de riesgo probabilistico (calidad del score). Esta dualidad es particularmente relevante porque aproxima mejor el escenario real de soporte a la decision, donde importa tanto ordenar pacientes por riesgo como estimar probabilidades razonables.

El dataset del reto integra variables estaticas y series temporales de las primeras 48 horas, con muestreo irregular y cobertura heterogenea por variable. Este diseño introdujo, desde su origen, retos de imputacion, agregacion temporal, tratamiento de ausencias y seleccion de modelos robustos. Por ello, el Challenge se consolidó como referencia para comparar pipelines que combinan ingenieria de variables clinicas, modelado probabilistico y evaluacion con metricas de discriminacion y calibracion.

En continuidad con ese marco, MIMIC-TRIAGE adopta la misma ventana temporal de 0-48h y el mismo objetivo clinico (mortalidad in-hospital), pero incorpora una evaluacion explicita orientada a triaje mediante metricas top-k, con foco en capacidad de captura de casos de alto riesgo en la parte prioritaria del ranking.

### 3.2.3 Enfoques tipicos reportados en el Challenge 2012

La literatura derivada del Challenge 2012 reporta una diversidad de estrategias: regresion logistica y variantes GLM, maquinas de soporte vectorial, random forests, modelos bayesianos y ensamblados. De forma general, los mejores resultados no dependieron solo del algoritmo, sino de la calidad del pipeline de preprocesado, de la representacion temporal y de la gestion del dato faltante.

Un punto de referencia importante es el baseline oficial distribuido con el reto, basado en una aproximacion bayesiana alrededor de SAPS_SCORE (por ejemplo en la implementacion de referencia physionet2012.m). Este baseline es util como ancla historica: conecta la tradicion de scores clinicos con formulaciones probabilisticas reproducibles, y deja claro que incluso modelos simples pueden ser competitivos cuando el pipeline esta bien especificado.

A partir de esos trabajos, se consolidan varias lecciones metodologicas aplicables a MIMIC-TRIAGE: (i) necesidad de particiones reproducibles, (ii) evaluacion consistente en test fijo para comparabilidad, (iii) cuidado con overfitting en espacios de alta dimensionalidad y (iv) reporte simultaneo de discriminacion y calibracion.

### 3.2.4 Modelos recientes para mortalidad UCI: arboles de gradiente y calibracion

En trabajos posteriores al Challenge, los modelos basados en arboles de decision ensamblados (Gradient Boosting, XGBoost, LightGBM, Random Forest) han mostrado desempeno robusto en prediccion de desenlaces en UCI, especialmente cuando se trabaja con variables tabulares heterogeneas, no linealidades e interacciones de alto orden. Su atractivo practico incluye manejo flexible de escalas, tolerancia a relaciones no monotonicas y buenos compromisos entre rendimiento y costo computacional.

De forma complementaria, se ha enfatizado que una AUC elevada no garantiza utilidad clinica por si sola. Por ello, la literatura reciente recomienda reportar Brier score, curvas de calibracion y, cuando procede, tecnicas de recalibracion (por ejemplo, Platt o isotonic) para asegurar que las probabilidades puedan interpretarse como riesgo utilizable. Este punto es central para sistemas de triaje, donde una probabilidad mal calibrada puede distorsionar la asignacion de prioridad.

Asimismo, la interpretabilidad ha pasado a primer plano. En modelos lineales se analizan coeficientes y signos; en modelos de arboles se emplean importancias por ganancia, permutation importance y enfoques explicativos como SHAP. Esta linea permite enlazar el comportamiento del modelo con conocimiento clinico y con hallazgos EDA, mejorando confianza, auditabilidad y adopcion.

### 3.2.5 Sintesis comparativa y gap que cubre MIMIC-TRIAGE en Hito 3

La evidencia disponible sugiere que no existe un unico algoritmo universalmente superior para todas las cohortes UCI: el rendimiento depende de la calidad del preprocesado, la representacion temporal, la gestion del desbalanceo y la validacion experimental. Aunque abundan resultados con metricas clasicas, es menos frecuente encontrar comparaciones sistematicas que integren simultaneamente: (1) discriminacion, (2) calibracion y (3) metricas operativas de priorizacion top-k definidas para triaje.

Ese es precisamente el gap que aborda este Hito 3: realizar una comparacion rigurosa, sobre un mismo pipeline de features agregadas 0-48h, entre modelos lineales y no lineales, reportando AUC-ROC, Brier, Recall@k y porcentaje de fallecidos en top-k. Con ello, el proyecto pasa de un baseline funcional (Hito 2) a una evaluacion cientifica reproducible orientada a decisiones clinicas de priorizacion.

## 3.3 Resultados esperables de la fase

### 3.3.1 Texto utilizable en el documento final
- Introduccion alineada con problema clinico, objetivo de triaje y formulacion probabilistica/ranking.
- Estado del arte estructurado en tres capas: scores clasicos, benchmark PhysioNet 2012 y modelos ML recientes con foco en calibracion.
- Cierre argumental con gap especifico y contribucion de Hito 3.

### 3.3.2 Tabla sugerida para el paper (resumen de literatura)

| Bloque | Que aporta a Hito 3 | Implicacion metodologica |
|---|---|---|
| APACHE II / SAPS / SOFA | Referencia clinica historica y comparabilidad | Mantener interpretabilidad y contexto de severidad |
| Challenge PhysioNet 2012 | Benchmark de mortalidad temprana 0-48h | Evaluacion comparable y protocolos reproducibles |
| Enfoques clasicos (GLM, SVM, RF, ensembles) | Base de modelos candidatos | Seleccion de familia de modelos para comparacion |
| Modelos de arboles recientes | Mejor captura de no linealidades e interacciones | Incluir GBM/XGBoost y analizar calibracion |
| Interpretabilidad moderna (importancias/SHAP) | Transparencia para adopcion clinica | Añadir analisis de variables relevantes |

### 3.3.3 Figuras sugeridas para acompañar esta fase
1. Diagrama conceptual de MIMIC-TRIAGE: datos 0-48h -> modelo -> probabilidad -> ranking clinico.
2. Esquema cronologico breve: scores clasicos -> Challenge 2012 -> modelos de arboles modernos.
3. Tabla visual de gap/reto: que evalua la literatura y que agrega el pipeline del proyecto (metricas top-k de triaje).

## 3.4 Checklist Fase 2
- [x] Introduccion redactada (contexto clinico + motivacion).
- [x] Estado del arte sobre acuity scores.
- [x] Estado del arte sobre PhysioNet 2012 y modelos tipicos.
- [x] Mencion de trabajos recientes con modelos de arbol y calibracion.
- [x] Identificado el gap que Hito 3 pretende cubrir.

## 3.5 Referencias base recomendadas para citar en el informe final
- PhysioNet/CinC Challenge 2012 (descripcion del reto y datos).
- Publicacion resumen de resultados del Challenge 2012.
- Documento de baseline oficial SAPS_SCORE (implementacion de referencia del reto).
- Literatura de mortalidad UCI con modelos de arboles y reporte de calibracion.
- Referencias clinicas de APACHE II, SAPS y SOFA como contexto de scores tradicionales.
