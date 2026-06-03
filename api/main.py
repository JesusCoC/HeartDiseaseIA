from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
import joblib
import pandas as pd
import warnings
import os

warnings.filterwarnings("ignore")

app = FastAPI()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'Heart_model.pkl')
SCALER_PATH = os.path.join(BASE_DIR, 'Heart_scaler.pkl')

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

@app.get("/")
def mostrar_pagina():
    ruta_html = os.path.join(os.path.dirname(BASE_DIR), 'index.html')
    return FileResponse(ruta_html)

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

    # NUEVA LÓGICA DE 4 NIVELES Y DIAGNÓSTICOS CLÍNICOS
    if probabilidad_enfermedad <= 25:
        nivel = "Riesgo Mínimo"
        recomendacion = "Mantenga su rutina actual. Se aconseja continuar con una dieta baja en sodio, realizar al menos 150 minutos de actividad física cardiovascular por semana y programar sus chequeos preventivos anuales estándar."
    elif probabilidad_enfermedad <= 50:
        nivel = "Riesgo Leve a Moderado"
        recomendacion = "Se sugiere reducir la ingesta de grasas saturadas y azúcares refinados. Incorpore ejercicio aeróbico de intensidad moderada sin picos de esfuerzo. Vigile su presión arterial mensualmente y considere solicitar un perfil lipídico completo en su próxima revisión."
    elif probabilidad_enfermedad <= 75:
        nivel = "Riesgo Elevado"
        recomendacion = "Es fundamental implementar cambios inmediatos: limite el consumo de sodio a menos de 2,000 mg diarios, evite el esfuerzo físico extenuante o de alto impacto sin supervisión. Se recomienda programar un electrocardiograma (ECG) de esfuerzo y una consulta de valoración a corto plazo."
    else:
        nivel = "Riesgo Crítico"
        recomendacion = "Requiere evaluación cardiológica prioritaria. Suspenda temporalmente actividades físicas de alto impacto. Adopte una dieta estricta cardioprotectora (tipo DASH) y establezca un monitoreo diario de su presión arterial y frecuencia cardíaca en reposo hasta ser evaluado por un especialista."

    return {
        "probabilidad": probabilidad_enfermedad,
        "nivel": nivel,
        "recomendacion": recomendacion
    }