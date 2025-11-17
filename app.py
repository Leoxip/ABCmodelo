import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import json

# =====================================================
# CONFIGURACIÓN
# =====================================================
st.set_page_config(
    page_title="Predicción Cardiovascular",
    page_icon="❤️",
    layout="wide"
)

st.title("❤️ Predicción de Riesgo Cardiovascular")


# =====================================================
# CARGAR MODELO
# =====================================================
MODEL_PATH = "Artefactos/v1/pipeline_MLP.joblib"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"No se encontró el modelo en: {MODEL_PATH}")
        st.stop()
    return joblib.load(MODEL_PATH)

model = load_model()


# =====================================================
# TABS
# =====================================================
tab1, tab2, tab3 = st.tabs(["🔮 Predicción", "📊 Gráficos", "📘 Interpretación"])


# =====================================================
# TAB 1 - PREDICCIÓN
# =====================================================
with tab1:

    st.header("🔮 Predicción de riesgo cardiovascular")

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Edad (años)", 18, 100, 50)
        height = st.number_input("Altura (cm)", 120, 220, 165)
        weight = st.number_input("Peso (kg)", 40.0, 200.0, 70.0)
        ap_hi = st.number_input("Presión sistólica (ap_hi)", 80, 250, 120)

    with col2:
        ap_lo = st.number_input("Presión diastólica (ap_lo)", 50, 200, 80)
        cholesterol = st.selectbox("Colesterol", ["Normal", "Medio", "Alto"])
        gluc = st.selectbox("Glucosa", ["Normal", "Elevada", "Muy Elevada"])
        smoke = st.selectbox("Fuma", ["No fuma", "Fuma"])
        alco = st.selectbox("Consume alcohol", ["No consume alcohol", "Consume alcohol"])
        active = st.selectbox("Actividad física", ["Activo", "Inactivo"])

    # =====================================================
    # CREAR DATA EXACTA QUE ESPERA EL MODELO
    # =====================================================
    input_data = pd.DataFrame({
        "id": [0],
        "age": [age * 365],
        "gender": ["Hombre"],  
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

    # ➕ Agregar columnas que el modelo espera
    input_data["age_years"] = input_data["age"] / 365
    input_data["BMI"] = input_data["weight"] / ((input_data["height"] / 100)**2)

    # =====================================================
    # BOTÓN DE PREDICCIÓN
    # =====================================================
    if st.button("Predecir riesgo", use_container_width=True):

        try:
            pred = model.predict(input_data)[0]
            proba = float(model.predict_proba(input_data)[0][1])

            # Resultado textual
            if pred == 1:
                st.error(f"⚠️ Riesgo cardiovascular — Probabilidad: {proba:.2f}")
            else:
                st.success(f"✅ Sin riesgo — Probabilidad: {proba:.2f}")

            # =====================================================
            # GRAFICO DE VELOCÍMETRO (GAUGE)
            # =====================================================
            fig, ax = plt.subplots(figsize=(5, 3))

            ax.axis("off")
            ax.annotate(
                "", xy=(0.5, 0), xytext=(0.5, -0.2),
                arrowprops=dict(arrowstyle="<-", lw=2)
            )

            # Barras del gauge
            colors = ["green", "yellow", "orange", "red"]
            thresholds = [0.25, 0.50, 0.75, 1.0]

            start = 0
            for c, t in zip(colors, thresholds):
                ax.barh(0, t - start, left=start, height=0.2, color=c)
                start = t

            # Aguja
            ax.plot([proba], [0.1], marker="v", markersize=12, color="black")
            ax.text(proba, 0.25, f"{proba:.2f}", ha="center")

            st.pyplot(fig)

            # =====================================================
            # INTERPRETACIÓN AUTOMÁTICA
            # =====================================================
            st.subheader("🧠 Interpretación automática")

            if proba < 0.25:
                st.success("✔ Riesgo muy bajo.")
            elif proba < 0.50:
                st.info("ℹ Riesgo bajo-moderado.")
            elif proba < 0.75:
                st.warning("⚠ Riesgo moderado.")
            else:
                st.error("❗ Riesgo alto. Se recomienda atención.")

        except Exception as e:
            st.error("Error durante la predicción.")
            st.code(str(e))


# =====================================================
# TAB 2 - GRÁFICOS DEL MODELO
# =====================================================
with tab2:

    st.header("📊 Gráficos del modelo entrenado")

    try:
        with open("Artefactos/v1/decision_policy.json") as f:
            dp = json.load(f)

        cm = np.array(dp["confusion_matrix"])
        labels = ["Sin riesgo", "Con riesgo"]

        # Matriz de confusión
        fig1, ax1 = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=labels, yticklabels=labels, ax=ax1)
        ax1.set_title("Matriz de Confusión")
        st.pyplot(fig1)

        # Métricas
        metrics = dp["test_metrics"]
        fig2, ax2 = plt.subplots()
        sns.barplot(x=list(metrics.keys()), y=list(metrics.values()), ax=ax2)
        ax2.set_title("Métricas del Modelo")
        plt.xticks(rotation=45)
        st.pyplot(fig2)

    except Exception as e:
        st.warning("No se pudieron cargar los gráficos")
        st.code(str(e))


# =====================================================
# TAB 3 - INTERPRETACIÓN
# =====================================================
with tab3:

    st.header("📘 Explicación de métricas")

    st.write("""
    - **Accuracy** → Precisión general del modelo.  
    - **Precision** → Qué tan correctas son las predicciones positivas.  
    - **Recall** → Capacidad para detectar casos con riesgo.  
    - **F1-score** → Balance entre precision y recall.  
    - **ROC-AUC** → Qué tan bien separa las clases.  
    """)

    try:
        st.json(dp["test_metrics"])
    except:
        st.info("No se pudieron cargar métricas.")
