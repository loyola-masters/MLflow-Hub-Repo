import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("sqlite:///mlflow.db")

client = MlflowClient()

model_name="Iris LR Model"
version_number=1
stage = "Staging"

# Cambiar el stage a "Staging", por ejemplo
client.transition_model_version_stage(
    name=model_name,
    version=version_number,
    stage=stage,
    archive_existing_versions=False
)

'''
Algunos valores de `stage` comunes son:
- "None" (por defecto)
- "Staging"
- "Production"
- "Archived"

El parámetro `archive_existing_versions` (por defecto `False`) indica si quieres archivar otras versiones de ese modelo que se encuentren en el mismo stage.
'''