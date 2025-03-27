import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, accuracy_score
from datetime import datetime
import numpy as np

# Establece el experimento
mlflow.set_experiment("iris")

# Nombre de la ejecución con timestamp
run_name = f"run_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

# Carga de datos
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Entrenamiento y tracking con MLflow
with mlflow.start_run(run_name=run_name):
    max_iter = 10  # Para simular iteraciones y ver convergencia
    model = LogisticRegression(
        max_iter=4,  # 4 iteraciones por ciclo
        warm_start=True,  # Para continuar el entrenamiento
        solver="saga",  # Necesario para warm_start y soporte multiclass
        multi_class="multinomial"
    )
    
    '''
    - Imprime métricas en consola durante el entrenamiento (en cada iteración)
    - Registra estas métricas en MLflow para observar la convergencia

    Dado que `LogisticRegression` de `scikit-learn` no expone las métricas por iteración
    directamente mediante `fit`, usaremos la opción solver='saga' o 'liblinear'
    junto con `warm_start=True` para simular el entrenamiento iterativo:

    - Ajusta `max_iter=4` dentro del bucle para controlar las iteraciones en cada epoch
    - Usa `warm_start=True` para continuar el entrenamiento sin reiniciar los pesos.
    - Muestra en consola y registra en MLflow el **accuracy** y el **log loss** en cada época (`step`).
    - Guarda el modelo final al terminar.
    '''
    
    mlflow.log_param("solver", "saga")
    mlflow.log_param("multi_class", "multinomial")
    mlflow.log_param("warm_start", True)
    mlflow.log_param("max_iter_total", max_iter)

    for epoch in range(1, max_iter + 1):
        model.fit(X_train, y_train)

        # Métricas
        y_pred_proba = model.predict_proba(X_test)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        loss = log_loss(y_test, y_pred_proba)

        print(f"Iteración {epoch}: Accuracy={acc:.4f}, LogLoss={loss:.4f}")

        # Reporte a MLflow
        mlflow.log_metric("accuracy", acc, step=epoch)
        mlflow.log_metric("log_loss", loss, step=epoch)

    # Guardar el modelo final
    mlflow.sklearn.log_model(model, "modelo_final")

    print(f"Run almacenado en: {mlflow.get_artifact_uri()}")
