import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import h5py
import requests
from tensorflow.keras.applications import Xception
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.python.keras.saving.hdf5_format import load_weights_from_hdf5_group
import contextlib

# Configuración de la página
st.set_page_config(page_title='Predicción de Retinopatía Diabética', page_icon='🩺', layout='wide')

# Estilo CSS personalizado
st.markdown("""
    <style>
        .reportview-container {
            background: #f0f2f6;
            padding: 2rem;
        }
        .sidebar .sidebar-content {
            background: #2c3e50;
            color: white;
        }
        h1 {
            color: #2980b9;
        }
        .stButton>button {
            background: #2980b9;
            color: white;
        }
    </style>
""", unsafe_allow_html=True)

# Título de la aplicación
st.title('Predicción de Retinopatía Diabética con Deep Learning')

# Descripción de la aplicación
st.write("""
    Esta aplicación utiliza un modelo de deep learning para predecir si una imagen muestra signos de retinopatía diabética.
    Simplemente sube una imagen y el modelo hará una predicción.
""")

# Descargar el modelo desde Google Drive si no existe
def download_model():
    model_url = 'http://server01.labs.org.pe:2005/Xception_diabetic_retinopathy_colab_v2.h5'
    output = 'Xception_diabetic_retinopathy_colab_v2.h5'
    if not os.path.exists(output):
        try:
            response = requests.get(model_url)
            with open(output, 'wb') as f:
                f.write(response.content)
            st.success("Modelo descargado correctamente.")
        except Exception as e:
            st.error(f"Error al descargar el modelo: {e}")

download_model()

# Verificación de carga de archivo
uploaded_file = st.file_uploader("Elige una imagen...", type=["jpg", "jpeg", "png"], label_visibility="hidden")

if uploaded_file is not None:
    # Definir el modelo
    target_size = (229, 229)
    base_model = Xception(weights=None, include_top=False, input_shape=target_size + (3,))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    # Ruta completa al modelo usando h5
    modelo_path = 'Xception_diabetic_retinopathy_colab_v2.h5'

    # Cargar los pesos del modelo
    try:
        with h5py.File(modelo_path, 'r') as f:
            load_weights_from_hdf5_group(f['model_weights'], model.layers)
        st.success("Modelo cargado correctamente.")
    except (UnicodeDecodeError, ValueError, OSError) as e:
        st.error(f"Error al cargar el modelo: {e}")
        model = None

    # Mostrar la imagen subida
    st.image(uploaded_file, output_format="auto", width=300, caption=None)

    if model is not None:
        # Preprocesamiento de la imagen para hacer la predicción
        img = image.load_img(uploaded_file, target_size=target_size)  # Ajusta según las dimensiones de entrada de tu modelo
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0

        # Redirigir la salida estándar y los mensajes de error a un objeto nulo
        with open(os.devnull, 'w') as devnull:
            with contextlib.redirect_stdout(devnull):
                with contextlib.redirect_stderr(devnull):
                    # Realizar la predicción
                    prediction = model.predict(img_array)

        # Mostrar resultados
        if prediction[0][0] > 0.5:
            st.error('El modelo predice que la imagen muestra signos de retinopatía diabética.')
        else:
            st.success('El modelo predice que la imagen no muestra signos de retinopatía diabética.')
