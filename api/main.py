from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import warnings
import os

warnings.filterwarnings("ignore")

app = FastAPI()

# Obtener la ruta absoluta del directorio actual (la carpeta api/)
# Esto evita errores de rutas cuando Vercel ejecuta el código en la nube
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'modelo_corazon.pkl')
SCALER_PATH = os.path.join(BASE_DIR, 'escalador_corazon.pkl')

try:
    modelo = joblib.load(MODEL_PATH)
    escalador = joblib.load(SCALER_PATH)
except Exception as e:
    print(f"Error cargando archivos: {e}")

class PacienteData(BaseModel):
    Age: float
    Sex: int
    ChestPainType: int
    RestingBP: float
    Cholesterol: float
    FastingBS: int
    RestingECG: int
    MaxHR: float
    ExerciseAngina: int
    Oldpeak: float
    ST_Slope: int

# Fíjate que la ruta ahora es /api/predecir
@app.post("/api/predecir")
def predecir_riesgo(datos: PacienteData):
    nombres_columnas = ["Age", "Sex", "ChestPainType", "RestingBP", "Cholesterol", "FastingBS", "RestingECG", "MaxHR", "ExerciseAngina", "Oldpeak", "ST_Slope"]
    datos_df = pd.DataFrame([[
        datos.Age, datos.Sex, datos.ChestPainType, datos.RestingBP, 
        datos.Cholesterol, datos.FastingBS, datos.RestingECG, 
        datos.MaxHR, datos.ExerciseAngina, datos.Oldpeak, datos.ST_Slope
    ]], columns=nombres_columnas)

    columnas_numericas = ["Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak"]
    datos_df[columnas_numericas] = escalador.transform(datos_df[columnas_numericas])

    probabilidades = modelo.predict_proba(datos_df)
    probabilidad_enfermedad = round(probabilidades[0][1] * 100, 1)

    if probabilidad_enfermedad < 30:
        recomendacion = "Riesgo Bajo. Mantenga un estilo de vida saludable."
    elif probabilidad_enfermedad < 65:
        recomendacion = "Riesgo Moderado. Se recomienda programar un chequeo médico de rutina."
    else:
        recomendacion = "Riesgo Alto. Por favor, consulte a un cardiólogo a la brevedad."

    return {
        "probabilidad": probabilidad_enfermedad,
        "recomendacion": recomendacion
    }