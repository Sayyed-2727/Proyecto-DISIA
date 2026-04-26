# 📘 GUÍA DE USO – Proyecto DISIA Hito 4  
## Sistema de Predicción de Mortalidad (MIMIC – Versión Simplificada 8 Features)

---

# ✅ 1️⃣ Cambios realizados en esta versión

Durante la corrección del sistema se realizaron los siguientes cambios importantes:

### 🔹 1. Modelo simplificado
Se creó un nuevo script:

```
src/train_simple.py
```

Este modelo:

- Usa solo **8 variables clínicas**
- Genera un modelo RandomForest
- Guarda un bundle con:
  - Modelo entrenado
  - Lista de features
  - Métricas (AUC, Accuracy)
  - Mapeo de variables

---

### 🔹 2. Corrección de rutas en Docker

Se corrigieron rutas absolutas dentro del contenedor:

Dataset:
```
/app/Hito 2/csvs/processed_features_48h_setA.csv
```

Modelo guardado en:
```
/app/models/best_mortality_model.pkl
```

Esto garantiza que:

✅ El modelo se sobrescriba correctamente  
✅ La API cargue el modelo actualizado  
✅ No vuelva a usar el modelo antiguo de 337 features  

---

### 🔹 3. Recarga correcta del modelo en la API

Después de regenerar el modelo, es necesario reiniciar la API:

```
docker compose restart mimic-api
```

Si no se reinicia, la API mantiene el modelo antiguo en memoria.

---

# 🚀 2️⃣ Cómo levantar el proyecto desde cero

Ubicación del proyecto:

```
Proyecto-DISIA/codigo
```

---

## ✅ Paso 1 – Entrar en la carpeta

```bash
cd Proyecto-DISIA/codigo
```

---

## ✅ Paso 2 – Apagar todo (si estaba corriendo)

```bash
docker compose down
```

---

## ✅ Paso 3 – Construir todo sin caché

```bash
docker compose build --no-cache
```

Esto:

- Construye API
- Construye Dashboard
- Construye Trainer

---

## ✅ Paso 4 – Levantar servicios

```bash
docker compose up -d
```

---

## ✅ Paso 5 – Verificar que todo está corriendo

```bash
docker ps
```

Debe aparecer:

- mimic_api
- mimic_dashboard
- mimic_trainer

---

# 🌐 3️⃣ Acceso al sistema

### 📊 Dashboard (Streamlit)

👉 http://localhost:8501

---

### 🔌 API (FastAPI)

👉 http://localhost:8000/docs

---

# 🧠 4️⃣ Cómo regenerar el modelo manualmente

Si quieres volver a entrenar el modelo:

```bash
docker compose build --no-cache train
docker compose up -d
```

Después:

```bash
docker compose restart mimic-api
```

---

# ⚠️ 5️⃣ Problemas comunes y solución

---

### ❌ Error: "El modelo espera 337 features"

Solución:

```bash
docker compose restart mimic-api
```

---

### ❌ Modelo no se sobreescribe

Verifica:

```bash
docker exec mimic_trainer ls -lh /app/models/
```

Debe cambiar la fecha del archivo.

---

### ❌ Contenedor mimic_trainer en Exited (1)

Ver logs:

```bash
docker logs mimic_trainer
```

---

# 🏥 6️⃣ Variables usadas por el modelo

El modelo actual usa solo:

1. Age
2. Heart Rate Mean
3. SysBP Mean
4. DiasBP Mean
5. Resp Rate Mean
6. Temperature Mean
7. SpO2 Mean
8. Glucose Mean

---

# 📌 7️⃣ Comportamiento del modelo

El modelo es un **RandomForest**, por lo que:

- No extrapola fuera del rango del dataset
- Valores extremos (ej: HR=1.000.000) no aumentan linealmente el riesgo
- Se recomienda validar inputs clínicamente realistas

---

# ✅ Estado actual del sistema

✅ Docker funcional  
✅ API funcional  
✅ Dashboard funcional  
✅ Modelo simplificado operativo  
✅ Persistencia correcta del modelo  
✅ Recarga correcta en API  

---

# 🎯 Proyecto listo para entrega y demostración

Sistema completamente operativo en entorno Docker.
