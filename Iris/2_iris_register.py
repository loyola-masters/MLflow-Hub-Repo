import mlflow

# Configurar conexión al tracking server (SQLite)
mlflow.set_tracking_uri("sqlite:///mlflow.db")

# Reemplaza con el run_id proporcionado por el script anterior
# run_id = "[PEGA_AQUÍ_EL_RUN_ID]"
run_id = "1d1bcbdf4f9e4f3fa6a2d3e449fa3881"

model_uri = f"runs:/{run_id}/model"

# Registrar modelo en el Model Registry
result = mlflow.register_model(model_uri=model_uri, name="Iris LR Model")

print(f"Modelo registrado: {result.name}, versión: {result.version}")

# ========================
#  COMPROBACIÓN DE ESTADO
# ========================
from mlflow.tracking import MlflowClient

# Comprobar el estado de la versión
client = MlflowClient()
model_version_info = client.get_model_version(name=result.name, version=result.version)
print("Estado del modelo registrado:", model_version_info.status)