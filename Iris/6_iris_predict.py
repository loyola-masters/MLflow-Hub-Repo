import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Configurar conexión al tracking server (SQLite)
mlflow.set_tracking_uri("sqlite:///mlflow.db")

# Cargar el modelo desde el registry
model = mlflow.sklearn.load_model("models:/Iris LR Model/1")

# Cargar dataset Iris y separar en X, y
iris = load_iris()
X, y = iris.data, iris.target
feature_names = iris.feature_names  # nombres de las cuatro features
class_names = iris.target_names     # ['setosa' 'versicolor' 'virginica']

# Dividir datos
_, X_test, _, y_test = train_test_split(X, y, random_state=42)

# Hacer predicción en las primeras 5 instancias de prueba
preds = model.predict(X_test[:5])

print("=== PREDICCIONES VS. VALORES REALES ===")
for idx, fila in enumerate(X_test[:5]):
    print(f"\n--- Fila {idx + 1} ---")
    
    # Imprimir cada feature con su nombre
    for f_idx, feat in enumerate(fila):
        print(f"{feature_names[f_idx]}: {feat}")
    
    # Mostrar predicción y valor real, tanto con su código numérico como con el nombre de la especie
    pred_code = preds[idx]
    real_code = y_test[idx]
    print(f"Predicción (cód.num): {pred_code} -> Especie predicha: {class_names[pred_code]}")
    print(f"Valor real (cód.num): {real_code} -> Especie real: {class_names[real_code]}")
