import mlflow

# Configurar conexión al tracking server (SQLite)
mlflow.set_tracking_uri("sqlite:///mlflow.db")

# Reemplaza con el run_id proporcionado por el script anterior
# run_id = "[PEGA_AQUÍ_EL_RUN_ID]"
run_id = "dd21ac4af35c49b6ab77330fad55e511"

model_uri = f"runs:/{run_id}/model"

# Registrar modelo en el Model Registry
result = mlflow.register_model(model_uri=model_uri, name="Iris LR Model")

print(f"Modelo registrado: {result.name}, versión: {result.version}")

# ========================
#  COMPROBACIÓN DE ESTADO
# ========================
from mlflow.tracking import MlflowClient

# Indicar la versión que se asignó
print(f"Modelo registrado: {result.name}, versión: {result.version}")

# Comprobar el estado de la versión
client = MlflowClient()
model_version_info = client.get_model_version(name=result.name, version=result.version)
print("Estado del modelo registrado:", model_version_info.status)