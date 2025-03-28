import requests
import json
from sklearn.datasets import load_iris

# Datos de prueba
X, _ = load_iris(return_X_y=True)
payload = {
    "inputs": X[:2].tolist()  # Primeras dos muestras
}

# Enviar POST request
response = requests.post(
    url="http://localhost:1234/invocations",
    headers={"Content-Type": "application/json"},
    data=json.dumps(payload)
)

print("Predicciones:", response.json())
