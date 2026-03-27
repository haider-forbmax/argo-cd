from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
import numpy as np

app = FastAPI()

# Train simple model at startup (no registry needed)
iris = load_iris()
X, y = iris.data, iris.target

model = LogisticRegression(max_iter=200)
model.fit(X, y)

class InputData(BaseModel):
    features: list[float]

@app.post("/predict")
def predict(data: InputData):
    features = np.array(data.features).reshape(1, -1)
    prediction = model.predict(features)[0]
    return {
        "class": int(prediction),
        "class_name": iris.target_names[prediction]
    }
