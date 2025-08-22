import os
import io
from typing import Tuple

import streamlit as st
from PIL import Image
import numpy as np
import cv2  # OpenCV (headless)
import pytesseract
from groq import Groq

# ---------------------------
# Configuración de la página
# ---------------------------
st.set_page_config(
    page_title="OCR + Groq (versión alternativa)",
    page_icon="🧩",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------
# Utilidades
# ---------------------------

def ensure_groq() -> Groq:
    """
    Obtiene el cliente de Groq tomando la API key de la barra lateral
    o de la variable de entorno GROQ_API_KEY.
    """
    key = st.session_state.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")
    if not key:
        st.warning("Falta GROQ_API_KEY (ponla en la barra lateral).")
        st.stop()
    return Groq(api_key=key)

def preprocess_image(pil_img: Image.Image) -> np.ndarray:
    """
    Preprocesamiento suave para mejorar el OCR:
    - A escala de grises
    - Filtro bilateral (reduce ruido conservando bordes)
    - Umbral adaptativo
    - Apertura morfológica ligera
    Devuelve una imagen binaria (numpy array) lista para Tesseract.
    """
    img = np.array(pil_img.convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    denoise = cv2.bilateralFilter(gray, d=7, sigmaColor=50, sigmaSpace=50)
    thr = cv2.adaptiveThreshold(
        denoise, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10
    )
    kernel = np.ones((2, 2), np.uint8)
    opened = cv2.morphologyEx(thr, cv2.MORPH_OPEN, kernel, iterations=1)
    return opened

def ocr_with_tesseract(pil_img: Image.Image, lang: str = "spa+eng") -> str:
    """
    Aplica preprocesamiento y extrae texto con Tesseract.
    """
    processed = preprocess_image(pil_img)
    # Para depurar: mostrar procesado en el panel izquierdo si se desea
    text = pytesseract.image_to_string(processed, lang=lang)
    return text.strip()

def call_groq_summarize(text: str, model: str = "llama-3.1-8b-instant") -> str:
    """
    Envía el texto OCR a Groq para:
    - corregir errores obvios de OCR
    - resumir / analizar
    """
    if not text:
        return "No se detectó texto en la imagen."
    client = ensure_groq()

    prompt = f"""
Eres un asistente que analiza texto obtenido por OCR.
1) Corrige solamente errores evidentes de OCR.
2) Resume en 5-8 viñetas claras (sin inventar).
3) Si el texto es una pregunta, respóndela brevemente.
4) Si son datos, organízalos en una lista.

TEXTO OCR:
---
{text}
---

Ahora, responde:
"""
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )
    return resp.choices[0].message.content.strip()

# ---------------------------
# Interfaz
# ---------------------------

st.title("🧩 OCR + Groq (implementación alternativa)")
st.caption("Versión distinta que usa Tesseract + OpenCV en lugar de EasyOCR.")

with st.sidebar:
    st.header("🔑 Configuración")
    api_key = st.text_input("GROQ_API_KEY", type="password", help="También puedes setearla como variable de entorno.")
    if api_key:
        st.session_state["GROQ_API_KEY"] = api_key

    st.markdown("—")
    st.subheader("🛠️ Opciones")
    lang = st.text_input("Idioma OCR (Tesseract)", value="spa+eng", help="Ej.: 'spa', 'eng', 'spa+eng'")
    model = st.selectbox("Modelo Groq", ["llama-3.1-8b-instant", "llama-3.1-70b-versatile"], index=0)

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("🖼️ Sube tu imagen")
    uploaded = st.file_uploader("PNG / JPG / JPEG", type=["png", "jpg", "jpeg"], label_visibility="collapsed")

    if uploaded is not None:
        pil_img = Image.open(uploaded).convert("RGB")
        st.image(pil_img, caption="Imagen cargada", use_column_width=True)

        # Vista del preprocesado (opcional)
        if st.toggle("Mostrar imagen preprocesada (para depurar OCR)", value=False):
            proc = preprocess_image(pil_img)
            st.image(proc, caption="Preprocesada", use_column_width=True, clamp=True)

        if st.button("✨ Extraer texto y analizar con Groq", type="primary", use_container_width=True):
            with st.spinner("🔍 Aplicando OCR..."):
                text = ocr_with_tesseract(pil_img, lang=lang)
                st.session_state["ocr_text"] = text

            with st.spinner("🧠 Consultando Groq..."):
                analysis = call_groq_summarize(text, model=model)
                st.session_state["groq_answer"] = analysis

with col2:
    st.subheader("💡 Resultados")
    if "ocr_text" in st.session_state:
        with st.expander("Texto extraído (OCR)"):
            st.text_area("Texto", st.session_state["ocr_text"], height=180)
        st.download_button(
            "⬇️ Descargar texto OCR",
            data=st.session_state["ocr_text"],
            file_name="ocr.txt",
            mime="text/plain",
            use_container_width=True
        )

    if "groq_answer" in st.session_state:
        st.markdown("#### Resumen / Análisis (Groq)")
        st.write(st.session_state["groq_answer"])
        st.download_button(
            "⬇️ Descargar análisis",
            data=st.session_state["groq_answer"],
            file_name="analisis_groq.txt",
            mime="text/plain",
            use_container_width=True
        )

st.info(
    "Nota: Para usar Tesseract debes tenerlo instalado en el sistema. "
    "En Ubuntu: `sudo apt-get install tesseract-ocr tesseract-ocr-spa`. "
    "En macOS: `brew install tesseract`. En Windows: instala el binario y agrega al PATH."
)
