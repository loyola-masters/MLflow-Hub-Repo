# CASE STUDY: Iris Classification (LAB7-Iris)

## 1. MLflow Tracking
Train the model:
```python
python .\1_iris_training.py
```
It is a modified version of `LogisticRegression` to expose metrics during the training process.

We will store thre results of the experiments in a SQLite database. To do so, first create an empty file (in the home folder) named `mlflow.db`. Then set its uri in the python script:
`mlflow.set_tracking_uri("sqlite:///mlflow.db")`

'0' is the 'Default' experiment. In our case we defined `mlflow.set_experiment("iris")`, that will assign `experiment_id = 1`

After running the script `1_iris_training.py`:
```
experiment_id = 1
run_id = 361398047b724cf0baec66856ac399da
```

To report metrics while training the model we define it as:
```
model = LogisticRegression(
        max_iter=4,  # 4 iteraciones por ciclo
        warm_start=True,  # Para continuar el entrenamiento desde el punto anterior (epoch)
        solver="saga",  # Necesario para warm_start y soporte multiclass
        multi_class="multinomial"
    )
```
Then we use:
- Nº epochs = 10
- Nº iterations per epoch = 4 ( = max_iter in `LogisticRegression` object above) 

Each epoch will report one point. Having 4 iterations we ensure a good degree of convergence in each one.

After the training finishes, the path to our model is:
`mlruns\[experiment_id]\[run_id]\artifacts\model\model.pkl`

The experiment data is stored in the SQLite database.

### Accediendo a Tracking UI
Since you setup the store `mlflow.db` (SQLite), you must add it to `mlflow ui` so that it can access experiments' data:
```powershell
mlflow ui --backend-store-uri "sqlite:///mlflow.db"
```
All data of the experiments will be saved here, instead of in `mlruns`
In any case, `model.pkl` will keep being stored in `mlruns\[experiment_id]\[run_id]\artifacts\model\model.pkl`

## 2. Run as a MLflow project
TO DO

## 3. Register the model in the database
`python .\2_iris_register.py`

Make sure that the final item in the path `model_uri = f"runs:/{run_id}/model"` coincides with model folder name as specified in the script `iris_training.py`, i.e. `mlflow.sklearn.log_model(model, "model")`

## 4. Cargar modelo y hacer predicciones
`python iris_predict.py`
 - Carga 5 muestras del dataset de test y proporciona las predicciones. junto con la comparación con los valores reales.


## 5. Exponer el modelo via API
```powershell
$Env:MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
mlflow models serve -m "models:/Iris LR Model/1" --host 0.0.0.0 --port 1234 --env-manager=local
```
La primera línea establece como variable de entorno MLFLOW_TRACKING_URI, que es la que apunta a la base de datos SQLite.
Las opciones para el comando `mlflow models serve` son las que se indican:

| Opción | Significado |
|--------|-------------|
| `-m "models:/MyModelName/1"` | Sirve el modelo del registro de modelos, versión 1 |
| `--port 1234` | Usa el puerto 1234 para la API |
| `--env-manager=local` | Usa el entorno Python actual |
| `--host 0.0.0.0` | Acepta conexiones externas (útil para Docker o red local) |

WARNING: *MLflow, a partir de ciertas versiones, intenta por defecto crear un entorno virtual para servir tu modelo usando la herramienta “pyenv”. Esto funciona bien en Linux/macOS, pero en Windows no suele estar instalada (o puede ser engorrosa de configurar). La solución más práctica es servir el modelo sin que MLflow intente crear un nuevo entorno. Con la opción `--env-manager=local` le indicamos que use el entorno Python actual*

---

## 6. Predicciones con llamadas a la API
Una vez esté corriendo, el endpoint de predicción será:
```
POST http://localhost:1234/invocations
```
Ejecutamos el script que hace esta petición:
`python .\7_iris_API_prediction.py`

Y obtenemos el siguiente resultado (5 muestras):
```
=== PREDICCIONES VS. VALORES REALES ===

--- Fila 1 ---
sepal length (cm): 6.1
sepal width (cm): 2.8
petal length (cm): 4.7
petal width (cm): 1.2
Predicción (cód.num): 1 -> Especie predicha: versicolor
Valor real (cód.num): 1 -> Especie real: versicolor

--- Fila 2 ---
sepal length (cm): 5.7
sepal width (cm): 3.8
petal length (cm): 1.7
petal width (cm): 0.3
Predicción (cód.num): 0 -> Especie predicha: setosa
Valor real (cód.num): 0 -> Especie real: setosa

--- Fila 3 ---
sepal length (cm): 7.7
sepal width (cm): 2.6
petal length (cm): 6.9
petal width (cm): 2.3
Predicción (cód.num): 2 -> Especie predicha: virginica
Valor real (cód.num): 2 -> Especie real: virginica

--- Fila 4 ---
sepal length (cm): 6.0
sepal width (cm): 2.9
petal length (cm): 4.5
petal width (cm): 1.5
Predicción (cód.num): 1 -> Especie predicha: versicolor
Valor real (cód.num): 1 -> Especie real: versicolor

--- Fila 5 ---
sepal length (cm): 6.8
sepal width (cm): 2.8
petal length (cm): 4.8
petal width (cm): 1.4
Predicción (cód.num): 1 -> Especie predicha: versicolor
Valor real (cód.num): 1 -> Especie real: versicolor
```

---
**NOTA:**

Si prefieres no usar el registro de modelos, sino directamente el `run_id`, también puedes servir desde la ruta local:
- Caso de un modelo registrado:
```bash
$Env:MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
mlflow models serve -m "runs:/0d694714335a4b1297672f55f06b391a/model" -p 1200 --env-manager=local
```
- Caso de un modelo no registrado (idem, no hay diferencia):
$Env:MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
mlflow models serve -m "runs:/7da7b04c13c442f9baa7042df9c9546b/model" -p 1200 --env-manager=local
---

## ANNEX: 'mlflow server' vs. 'mlflow models serve'
- **`mlflow server`** launches the **MLflow Tracking Server** (the UI + endpoints for experiments, runs, artifacts, and the model registry).
- **`mlflow models serve`** launches a **prediction (inference) server** for a specific model, exposing an `/invocations` endpoint to send data and receive predictions.

More in depth:

**1) `mlflow server`**
- Starts the MLflow Tracking Server which provides:
  - A **web UI** you can open in your browser to explore experiments, runs, parameters, metrics, and models.
  - A **REST API** to log runs and artifacts, register models, etc.
  - The possibility to configure a backend store (e.g., SQLite, MySQL, PostgreSQL) for experiment tracking and a default artifact store (local filesystem, S3, etc.) for run artifacts.
- Example:
  ```bash
  mlflow server \
    --backend-store-uri sqlite:///mlflow.db \
    --default-artifact-root ./mlruns \
    --host 0.0.0.0 \
    --port 5000
  ```
- Access the UI at: [http://localhost:5000](http://localhost:5000).

**2) `mlflow models serve`**

- Starts a **model scoring server** (inference server) for **one** specific model.
- Exposes a **single** endpoint at `POST /invocations` for making predictions in real time.
- Useful if you want to quickly serve a model in a development or test setting without writing your own API in Flask or FastAPI.
- Example:
  ```bash
  mlflow models serve -m "models:/MyModel/1" --port 1234 --env-manager=local
  ```
- After it’s running, send prediction requests to `POST http://localhost:1234/invocations` with data in JSON format.


Think of `mlflow server` as the **central repository (plus UI)** for your MLflow runs and model registry, whereas `mlflow models serve` is a **lightweight inference service** for a single model.

### Serving models from `mlflow server`
Although `mlflow models serve` allows you to expose a model, `mlflow server` approach decouples server configuration from model selection, allowing you to serve multiple models simultaneously on different ports while maintaining a single tracking server instance.

To select specific models when using the MLflow tracking server, you configure the model URI during deployment rather than at server startup. Here's how to manage model selection:

**Model Serving Workflow**
1. **Start MLflow Tracking Server** (as shown in your command):
```bash
mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlruns \
  --host 0.0.0.0 \
  --port 5000
```

2. **Reference Models Using URI Schemes**  
When deploying, use these URI patterns to select models:

| Model Source | URI Format | Example |
|--------------|------------|---------|
| Direct Run Artifact | `runs://` | `runs:/0d694714335a4b.../model` |
| Model Registry | `models://` | `models:/wine-quality/1` |
| Local File Path | `file:///` | `file:///C:/mlruns/.../model` |


**For Registered Models**:
```python
from mlflow.tracking import MlflowClient

client = MlflowClient()
model_version = client.get_latest_versions("wine-quality", stages=["Production"])[0]
model_uri = f"models:/{model_version.name}/{model_version.version}"
```

**For Direct Run References**:
```bash
mlflow models serve -m "runs:/0d694714335a4b1297672f55f06b391a/model" --port 5001
```

**Model Selection Best Practices**
1. Registry vs Direct References
   - Use `models:/` URIs for production (versioned, stage-managed models)
   - Use `runs:/` URIs for experimental/testing models

2. Windows Path Considerations
```bash
# Encode spaces and special characters
mlflow models serve -m "file:///C:/Users/.../Mi%20unidad/.../model"
```

3. Validation Before Serving
```python
from mlflow.models import validate_serving_input
validate_serving_input(model_uri, serving_example)
```

**Server Configuration Tips**

For high-volume model access, consider using separate `--artifact-only` servers:
```bash
# Primary server
mlflow server --no-serve-artifacts ...

# Artifact server
mlflow server --artifacts-only ...
```
When using SQLite backend, ensure write permissions to the database file

