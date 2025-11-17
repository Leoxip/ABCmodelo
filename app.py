import streamlit as st
import pandas as pd
import joblib
import os

# ==========================
# Configuración de la página
# ==========================
st.set_page_config(page_title="Predicción Cardiovascular", page_icon="❤️")

st.title("❤️ Predicción de Riesgo Cardiovascular")
st.write("Modelo de clasificación MLP entrenado por Mayra.")

# ==========================
# Cargar modelo
# ==========================
ARTIFACT_PATH = "Artefactos/v1/pipeline_MLP.joblib"


@st.cache_resource
def load_model():
    # Verifica existencia del archivo
    if not os.path.exists(ARTIFACT_PATH):
        st.error(f"❌ No se encontró el modelo en: {ARTIFACT_PATH}")
        st.stop()

    try:
        model = joblib.load(ARTIFACT_PATH)
        return model
    except Exception as e:
        st.error("❌ Error al cargar el modelo. Revisa compatibilidad de versiones.")
        st.code(str(e))
        st.stop()


model = load_model()

st.header("Ingrese los datos del paciente")

# ==========================
# Inputs del usuario
# ==========================

age = st.number_input("Edad (años)", min_value=18, max_value=100, value=50)
height = st.number_input("Altura (cm)", min_value=120, max_value=220, value=165)
weight = st.number_input("Peso (kg)", min_value=40, max_value=200, value=70)
ap_hi = st.number_input("Presión sistólica (ap_hi)", min_value=80, max_value=250, value=120)
ap_lo = st.number_input("Presión diastólica (ap_lo)", min_value=50, max_value=200, value=80)

cholesterol = st.selectbox("Colesterol", ["Normal", "Medio", "Alto"])
gluc = st.selectbox("Glucosa", ["Normal", "Elevada", "Muy Elevada"])
smoke = st.selectbox("Fuma", ["No fuma", "Fuma"])
alco = st.selectbox("Consume alcohol", ["No consume alcohol", "Consume alcohol"])
active = st.selectbox("Actividad física", ["Activo", "Inactivo"])

# ==========================
# Preparar DataFrame
# ==========================

input_data = pd.DataFrame({
    "age": [age * 365],    # Tu dataset usa edad en días
    "height": [height],
    "weight": [weight],
    "ap_hi": [ap_hi],
    "ap_lo": [ap_lo],
    "cholesterol": [cholesterol],
    "gluc": [gluc],
    "smoke": [smoke],
    "alco": [alco],
    "active": [active]
})

# ==========================
# Predicción
# ==========================
if st.button("Predecir riesgo"):
    
    try:
        pred = model.predict(input_data)[0]

        # Mostrar predicción
        if pred == 1:
            st.error("⚠️ El modelo predice: **Con riesgo cardiovascular**")
        else:
            st.success("✅ El modelo predice: **Sin riesgo cardiovascular**")

        # Probabilidades
        try:
            prob = model.predict_proba(input_data)[0][1]
            st.write(f"Probabilidad estimada: **{prob:.2f}**")
        except:
            st.info("El modelo no soporta predict_proba().")

    except Exception as e:
        st.error("❌ Error durante la predicción.")
        st.code(str(e))
