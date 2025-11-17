import streamlit as st
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ==========================
# Configuración de la página
# ==========================
st.set_page_config(page_title="Predicción Cardiovascular", page_icon="❤️", layout="wide")

st.title("❤️ Predicción de Riesgo Cardiovascular")
st.write("Aplicación web con modelo MLP entrenado por Mayra.")

# ==========================
# Cargar modelo
# ==========================
ARTIFACT_PATH = "Artefactos/v1/pipeline_MLP.joblib"

@st.cache_resource
def load_model():
    if not os.path.exists(ARTIFACT_PATH):
        st.error(f"❌ No se encontró el modelo en: {ARTIFACT_PATH}")
        st.stop()
    return joblib.load(ARTIFACT_PATH)

model = load_model()


# ==============================================
#             CREAR TABS (pestañas)
# ==============================================
tab1, tab2, tab3 = st.tabs([
    "🔮 Predicción",
    "📊 Gráficos del Modelo",
    "📘 Interpretación"
])


# ============================================================
#                       TAB 1 – PREDICCIÓN
# ============================================================
with tab1:
    st.header("🔮 Predicción de riesgo")

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Edad (años)", min_value=18, max_value=100, value=50)
        height = st.number_input("Altura (cm)", min_value=120, max_value=220, value=165)
        weight = st.number_input("Peso (kg)", min_value=40, max_value=200, value=70)
        ap_hi = st.number_input("Presión sistólica (ap_hi)", min_value=80, max_value=250, value=120)

    with col2:
        ap_lo = st.number_input("Presión diastólica (ap_lo)", min_value=50, max_value=200, value=80)
        cholesterol = st.selectbox("Colesterol", ["Normal", "Medio", "Alto"])
        gluc = st.selectbox("Glucosa", ["Normal", "Elevada", "Muy Elevada"])
        smoke = st.selectbox("Fuma", ["No fuma", "Fuma"])
        alco = st.selectbox("Consume alcohol", ["No consume alcohol", "Consume alcohol"])
        active = st.selectbox("Actividad física", ["Activo", "Inactivo"])

    # Preparar entrada
    input_data = pd.DataFrame({
        "age": [age * 365],
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

    if st.button("Predecir riesgo"):
        pred = model.predict(input_data)[0]
        proba = model.predict_proba(input_data)[0][1]

        if pred == 1:
            st.error(f"⚠️ Riesgo cardiovascular — probabilidad {proba:.2f}")
        else:
            st.success(f"✅ Sin riesgo — probabilidad {proba:.2f}")


# ============================================================
#               TAB 2 – GRÁFICOS DEL MODELO
# ============================================================
with tab2:
    st.header("📊 Análisis Visual del Modelo")

    # Mostrar matriz de confusión si existe el archivo
    try:
        import json
        with open("Artefactos/v1/decision_policy.json") as f:
            dp = json.load(f)

        cm = np.array(dp["confusion_matrix"])
        labels = ["Sin riesgo", "Con riesgo"]

        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=labels, yticklabels=labels, ax=ax)
        ax.set_title("Matriz de Confusión")
        st.pyplot(fig)

        # Gráfico de barras de métricas
        metrics = dp["test_metrics"]
        fig2, ax2 = plt.subplots()
        sns.barplot(x=list(metrics.keys()), y=list(metrics.values()), ax=ax2)
        plt.xticks(rotation=45)
        ax2.set_title("Métricas del Modelo")
        st.pyplot(fig2)

    except Exception as e:
        st.warning("⚠ No se pudieron cargar los gráficos del modelo.")
        st.code(str(e))


# ============================================================
#               TAB 3 – INTERPRETACIÓN DEL MODELO
# ============================================================
with tab3:
    st.header("📘 Interpretación de Resultados")

    st.subheader("🔍 Lectura de métricas")
    st.write("""
    - **Accuracy** indica el porcentaje de aciertos totales.  
    - **Precision** mide cuántas predicciones positivas fueron correctas.  
    - **Recall** mide la capacidad del modelo para detectar casos con riesgo.  
    - **F1-score** combina precisión y recall.  
    - **ROC-AUC** mide qué tan bien separa clases.  
    """)

    try:
        st.subheader("📈 Métricas del modelo")
        st.json(dp["test_metrics"])

    except:
        st.warning("No se encontró el archivo de métricas.")
