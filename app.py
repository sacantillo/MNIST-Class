import streamlit as st
import numpy as np
import pickle, cv2
from PIL import Image
import pandas as pd
import plotly.graph_objects as go
import os

# Etiquetas de Fashion MNIST
CLASS_NAMES = ['Dress',
 'Angle boot',
 'T-shirt',
 'Sandal',
 'Trouser',
 'Shirt',
 'Pullover',
 'Coat',
 'Bag',
 'Sneaker']
st.set_page_config(layout="wide")
st.title("Clasificador Fashion MNIST con modelo de aprendizaje precargado")


# Cargar modelo desde archivo local
@st.cache_resource
def load_model():
    with open("modelo.pkl", "rb") as f:
        return pickle.load(f)

def preprocess_uploaded_image(uploaded_file):
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError("No se pudo leer la imagen.")
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    img_norm = cv2.normalize(img_gray, None, 0, 255, cv2.NORM_MINMAX)
    return img_norm

# Cargar modelo
if not os.path.exists("modelo.pkl"):
    st.error("No se encontró el archivo 'modelo.pkl'. Por favor colócalo en el mismo directorio.")
else:
    model = load_model()
    st.success("Modelo cargado correctamente.")

    uploaded_file = st.file_uploader("Sube una imagen (preferiblemente de 28x28 px)", type=["png", "jpg", "jpeg"])
    if uploaded_file:
        try:
            img_processed = preprocess_uploaded_image(uploaded_file)
            img_resized = cv2.resize(img_processed, (28, 28))
            img_flat = img_resized.flatten().reshape(1, -1)
            prediction = model.predict(img_flat)[0]

            # Probabilidades
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(img_flat)[0]
                df_proba = pd.DataFrame({
                    "Clase": CLASS_NAMES,
                    "Probabilidad (%)": proba * 100
                }).sort_values("Probabilidad (%)", ascending=False)

                col1, col2 = st.columns([1, 2])

                with col1:
                    st.image(Image.fromarray(img_resized), caption="Imagen preprocesada (28x28, normalizada)", width=150)
                    st.subheader(f"Predicción: {CLASS_NAMES[prediction]}")

                with col2:
                    fig = go.Figure(go.Bar(
                        x=df_proba["Clase"],
                        y=df_proba["Probabilidad (%)"],
                        text=df_proba["Probabilidad (%)"].map("{:.2f}%".format),
                        textposition='auto'
                    ))

                    fig.update_layout(
                        title="Probabilidad de cada clase",
                        yaxis=dict(
                            title="Probabilidad (%)",
                            range=[0, 100],
                            ticksuffix="%",
                            showgrid=True
                        ),
                        xaxis=dict(title="Clase")
                    )

                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Este modelo no soporta probabilidades.")
        except Exception as e:
            st.error(f"Ocurrió un error al procesar la imagen: {e}")
