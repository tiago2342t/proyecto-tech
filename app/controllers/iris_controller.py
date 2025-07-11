import joblib
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from io import BytesIO
from PIL import Image
import base64
from fastapi import APIRouter, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import pandas as pd
import os


# Crear el enrutador para las peticiones
router = APIRouter()

# Inicializar el objeto Jinja2Templates globalmente para el renderizado
templates = Jinja2Templates(directory="app/templates")

model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),  '..', 'models', 'kmeans_model.pkl')

# Cargar el modelo desde la ruta absoluta
model = joblib.load(model_path)
iris = load_iris()

def predict_iris(sepal_length: float, sepal_width: float, petal_length: float, petal_width: float):
    """Recibe las variables y devuelve la predicción del modelo"""
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(input_data)
    predicted_class = iris.target_names[prediction][0]
    return predicted_class

def generate_plot(sepal_length: float, sepal_width: float, petal_length: float, petal_width: float):
    """Genera una matriz de dispersión usando seaborn y marca el punto de entrada del usuario en todas las gráficas"""
    # Convertir los datos Iris a un DataFrame para usar con seaborn
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

    # Crear la matriz de dispersión con seaborn
    sns.set(style="ticks")
    g = sns.pairplot(df, hue="species", markers=["o", "s", "D"], palette="husl")

    # Agregar el punto de entrada del usuario en todas las gráficas
    for ax in g.axes.flatten():
        ax.scatter(sepal_length, sepal_width, color='black', label="Entrada del Usuario", edgecolors='white', s=100, zorder=5)

    # Leyenda
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc='upper right')

    # Guardar la gráfica como imagen en buffer
    buf = BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    img = Image.open(buf)
    img_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    
    # Cerrar la figura para liberar memoria
    plt.close(g.figure)

    return img_base64

@router.get("/", response_class=HTMLResponse)
async def form(request: Request):
    """Renderiza el formulario donde el usuario puede ingresar datos"""
    return templates.TemplateResponse("index.html", {"request": request})

@router.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, sepal_length: float = Form(...), sepal_width: float = Form(...),
                  petal_length: float = Form(...), petal_width: float = Form(...)):
    """Recibe las variables y devuelve la predicción y gráfico"""
    
    # Realizar la predicción usando el controlador
    predicted_class = predict_iris(sepal_length, sepal_width, petal_length, petal_width)

    # Generar la gráfica usando el controlador
    img_base64 = generate_plot(sepal_length, sepal_width, petal_length, petal_width)

    # Renderizar resultados en HTML
    return templates.TemplateResponse("result.html", {"request": request, "prediction": predicted_class, "image_base64": img_base64})
