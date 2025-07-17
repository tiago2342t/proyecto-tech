from fastapi import APIRouter, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import numpy as np
from app.main import model, le_ciudad, le_ubicacion

router = APIRouter()
templates = Jinja2Templates(directory="app/templates")

@router.get("/", response_class=HTMLResponse)
async def formulario(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@router.post("/predict/apartamento", response_class=HTMLResponse)
async def predecir_apartamento(
    request: Request,
    ciudad: str = Form(...),
    ubicación: str = Form(...),
    baños: int = Form(...),
    habitaciones: int = Form(...),
    garages: int = Form(...),
    area: float = Form(...),
    estrato: int = Form(...)
):
    try:
        ciudad_cod = le_ciudad.transform([ciudad])[0]
    except ValueError:
        ciudad_cod = 0
    try:
        ubicacion_cod = le_ubicacion.transform([ubicación])[0]
    except ValueError:
        ubicacion_cod = 0

    entrada = np.array([[ciudad_cod, ubicacion_cod, baños, habitaciones, garages, area, estrato]])
    prediccion = model.predict(entrada)[0]

    return templates.TemplateResponse("result.html", {
        "request": request,
        "prediction": f"${prediccion:,.0f}",
        "image_base64": ""
    })

@router.post("/predict/casa", response_class=HTMLResponse)
async def predecir_casa(
    request: Request,
    ciudad: str = Form(...),
    ubicación: str = Form(...),
    baños: int = Form(...),
    habitaciones: int = Form(...),
    garages: int = Form(...),
    area: float = Form(...),
    estrato: int = Form(...)
):
    try:
        ciudad_cod = le_ciudad.transform([ciudad])[0]
    except ValueError:
        ciudad_cod = 0
    try:
        ubicacion_cod = le_ubicacion.transform([ubicación])[0]
    except ValueError:
        ubicacion_cod = 0

    entrada = np.array([[ciudad_cod, ubicacion_cod, baños, habitaciones, garages, area, estrato]])
    prediccion = model.predict(entrada)[0]

    return templates.TemplateResponse("result.html", {
        "request": request,
        "prediction": f"${prediccion:,.0f}",
        "image_base64": ""
    })