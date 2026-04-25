# Hito 4

## Objetivo
Convertir el notebook "california_housing_regression" en un despliegue real.

## Organización
El ejemplo tiene dos variantes:
- Modo manual: todos los dicheros que no contienen "ml" o "mlflow" en su nombre.
- Con MLFlow: los ficheros contienen "mlflow" o "ml"

# Ejecución
## Manual
```bash
docker compose up train-manual
docker compose up api-manual
```

Acceder a http://localhost.8000/docs para ver la documentación de la api y realizar inferencias de ejemplo.

## MLFlow
```bash
docker compose -f docker-compose-mlflow.yml --profile training up retrain-job
```

Acceder a la web: localhost:5001 y registrar un modelo con el nombre California_Housing_Prod

```bash
docker compose -f docker-compose-mlflow.yml up api-serving
```

Acceder a http://localhost.8000/docs para ver la documentación de la api y realizar inferencias de ejemplo.