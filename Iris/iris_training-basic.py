import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from datetime import datetime

# (Opcional) Cambia la ruta si quieres guardar en otra carpeta
# mlflow.set_tracking_uri("file:///ruta/completa/a/mi/mlflow")

# Define experiment name. Skip here if running from a MLflow Project
mlflow.set_experiment("iris")

# Generate run name with timestamp
run_name = f"run_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    
# Carga datos
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Entrena y registra
with mlflow.start_run(run_name=run_name):
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)

    # Registra modelo localmente
    mlflow.sklearn.log_model(model, "modelo")

    # Log parámetros y métricas
    mlflow.log_param("max_iter", 200)
    mlflow.log_metric("accuracy", model.score(X_test, y_test))

    print(f"Run almacenado en: {mlflow.get_artifact_uri()}")
