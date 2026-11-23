from fastapi import APIRouter, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from pathlib import Path #se utiliza para trabajar con rutas de archivos y directorios de manera más conveniente y legible
import joblib #se utiliza para cargar modelos de machine learning previamente entrenados y guardados en archivos
import pandas as pd
import json  #se utiliza para trabajar con datos en formato JSON, como cargar configuraciones o datos de entrada para la predicción
# Cargar el modelo de machine learning desde un archivo .pkl
carpeta_modelos = Path(__file__).resolve().parents[1] / "models"

with (carpeta_modelos / "opciones_viviendas.json").open(encoding="utf-8") as archivo:
    opciones = json.load(archivo)
    
with (carpeta_modelos / "anuncios_comparables.json").open(encoding="utf-8") as archivo:
    anuncios_comparables = json.load(archivo)

model = joblib.load(carpeta_modelos / "modelo_viviendas.pkl")

router = APIRouter()
templates = Jinja2Templates(directory="app/templates")

# Cargar los anuncios comparables desde un archivo JSON
def buscar_comparables(ciudad, tipo_propiedad, ubicacion, area):
    def normalizar(texto):
        return str(texto).strip().casefold()

    sector = normalizar(ubicacion)

    if sector in ("", "no especificado"):
        return []

    coincidencias = [
        anuncio
        for anuncio in anuncios_comparables
        if normalizar(anuncio["Ciudad"]) == normalizar(ciudad)
        and normalizar(anuncio["Tipo Propiedad"]) == normalizar(tipo_propiedad)
        and normalizar(anuncio["Ubicacion Principal"]) == sector
        and abs(anuncio["Area Construida"] - area) <= area * 0.20
    ]

    # Una diferencia de 2 m² se considera más cercana que una de 10 m².
    coincidencias.sort(
        key=lambda anuncio: abs(anuncio["Area Construida"] - area)
    )

    return [
        {
            "area": anuncio["Area Construida"],
            "precio": anuncio["Precio"],
            "precio_m2": anuncio["Precio"] / anuncio["Area Construida"]
        }
        for anuncio in coincidencias[:5]
    ]

# Definición de rutas para la aplicación FastAPI
@router.get("/", response_class=HTMLResponse)
async def formulario(request: Request):
    return templates.TemplateResponse(
    request=request,
    name="index.html",
    context={"opciones": opciones}
)

@router.post("/predict/casa", response_class=HTMLResponse)
async def predecir_vivienda(
    request: Request,
    tipo_propiedad: str = Form(...),
    ciudad: str = Form(...),
    ubicación: str = Form(...),
    baños: int = Form(...),
    habitaciones: int = Form(...),
    garages: int = Form(...),
    area: float = Form(...),
    estrato: int = Form(...),
    antiguedad: str = Form(...),
):
    # Crear un DataFrame con los datos de entrada para la predicción
    entrada = pd.DataFrame([{
        "Tipo Propiedad": tipo_propiedad,
        "Ciudad": ciudad,
        "Ubicacion Principal": ubicación,
        "Banos": baños,
        "Habitaciones": habitaciones,
        "Garages": garages,
        "Area Construida": area,
        "Estrato": estrato,
        "Antiguedad": antiguedad
    }])
    prediccion = model.predict(entrada)[0]
    
    comparables = buscar_comparables(
        ciudad=ciudad,
        tipo_propiedad=tipo_propiedad,
        ubicacion=ubicación,
        area=area
    )

    text_habitaciones = f"{habitaciones} {'habitación' if habitaciones == 1 else 'habitaciones'}"
    text_banos = f"{baños} {'baño' if baños == 1 else 'baños'}"

    if garages == 0:
        text_garajes = "Sin garaje"
    else:
        text_garajes = f"{garages} {'garaje' if garages == 1 else 'garajes'}"

    precio_m2 = "No disponible"
    if area > 0:
        importe_m2 = f"${prediccion / area:,.0f}".replace(",", ".")
        precio_m2 = f"{importe_m2} COP/m²"

        return templates.TemplateResponse(
        request=request,
        name="result.html",
        context={
            "prediction": f"${prediccion:,.0f}",
            "image_base64": "",
            "vivienda": {
                "tipo": tipo_propiedad,
                "ciudad": ciudad,
                "barrio": ubicación,
                "area": str(area).removesuffix(".0").replace(".", ","),
                "habitaciones": text_habitaciones,
                "banos": text_banos,
                "garajes": text_garajes,
                "estrato": estrato,
                "antiguedad": antiguedad,
            },
            "precio_m2": precio_m2,
            "comparables": comparables,
        },
)