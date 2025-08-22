import os
import io
from typing import List, Tuple

import streamlit as st
from groq import Groq
from PIL import Image, ImageOps
import easyocr
import numpy as np

# ---------------------------
# Config de página
# ---------------------------
st.set_page_config(
    page_title="OCR + Groq (Streamlit Cloud friendly)",
    page_icon="🧩",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------
# Utilidades
# ---------------------------

@st.cache_resource
def load_easyocr_reader(langs: List[str]):
    # Forzamos CPU para compatibilidad en Streamlit Cloud
    return easyocr.Reader(langs, gpu=False)

def ensure_groq() -> Groq:
    key = st.session_state.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")
    if not key:
        st.error("Falta GROQ_API_KEY. Ingresa tu key en la barra lateral.")
        st.stop()
    return Groq(api_key=key)

def normalize_image(pil_img: Image.Image) -> Image.Image:
    """
    Pequeño preprocesamiento:
    - Convertir a RGB
    - AutoOrient (EXIF) para corregir rotaciones
    - Aumentar contraste moderadamente
    """
    img = pil_img.convert("RGB")
    img = ImageOps.exif_transpose(img)
    # Opcional: convertir a escala de grises para EasyOCR no suele ser necesario
    return img

def easyocr_extract_text(pil_img: Image.Image, reader: easyocr.Reader) -> Tuple[str, list]:
    """
    Devuelve texto unido y resultados crudos (lista de [bbox, text, conf]).
    """
    arr = np.array(pil_img)
    result = reader.readtext(arr)
    texto = " ".join([r[1] for r in result]).strip()
    return texto, result

def call_groq(text: str, mode: str, model: str, temperature: float = 0.2) -> str:
    """
    Envía a Groq el texto OCR para:
    - Resumen
    - Análisis/organización de datos
    - Q&A si detecta pregunta
    """
    if not text:
        return "No se detectó texto en la imagen."
    client = ensure_groq()

    if mode == "Resumen":
        prompt = f"""Eres un asistente que analiza texto obtenido por OCR.
1) Corrige solo errores evidentes de OCR.
2) Resume en 5–8 viñetas claras, sin inventar.

TEXTO OCR:
---
{text}
---

Ahora responde con viñetas:"""
    else:  # "Analizar / Organizar"
        prompt = f"""Del siguiente texto extraído por OCR:
- Corrige errores evidentes.
- Si son datos, organízalos en una tabla de Markdown o lista estructurada.
- Si es una pregunta, respóndela brevemente.
- Si es un texto largo, resume puntos clave.

TEXTO:
---
{text}
---

Responde:"""

    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )
    return resp.choices[0].message.content.strip()

# ---------------------------
# UI
# ---------------------------

st.title("🧩 OCR + Groq (compatible con Streamlit Cloud)")
st.caption("Usa EasyOCR (sin binarios del sistema) y Groq para resumir/analizar el texto extraído.")

with st.sidebar:
    st.header("🔑 Configuración")
    api_key = st.text_input("GROQ_API_KEY", type="password", help="También puedes setearla como variable de entorno.")
    if api_key:
        st.session_state["GROQ_API_KEY"] = api_key

    st.subheader("🛠️ Opciones")
    langs_str = st.text_input("Idiomas OCR (EasyOCR)", value="es,en", help="Separados por coma. Ej: es,en")
    langs = [s.strip() for s in langs_str.split(",") if s.strip()]
    mode = st.radio("Modo", ["Resumen", "Analizar / Organizar"], index=0)
    model = st.selectbox("Modelo Groq", ["llama-3.1-8b-instant", "llama-3.1-70b-versatile"], index=0)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.05)

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("🖼️ Sube tu imagen")
    up = st.file_uploader("PNG/JPG/JPEG", type=["png", "jpg", "jpeg"], label_visibility="collapsed")
    if up:
        img = Image.open(up)
        st.image(img, caption="Imagen cargada", use_column_width=True)

        if st.button("✨ Ejecutar OCR + Groq", type="primary", use_container_width=True):
            # Cargar OCR (cacheado)
            try:
                reader = load_easyocr_reader(langs if langs else ["es", "en"])
            except Exception as e:
                st.error(f"No se pudo cargar EasyOCR: {e}")
                st.stop()

            with st.spinner("🔍 Extrayendo texto (EasyOCR)…"):
                img_norm = normalize_image(img)
                text, raw = easyocr_extract_text(img_norm, reader)
                st.session_state["ocr_text"] = text
                st.session_state["ocr_raw"] = raw

            with st.spinner("🧠 Analizando con Groq…"):
                out = call_groq(text, mode=mode, model=model, temperature=temperature)
                st.session_state["groq_out"] = out

with col2:
    st.subheader("💡 Resultados")
    if "ocr_text" in st.session_state:
        with st.expander("Texto OCR"):
            st.text_area("Texto", st.session_state["ocr_text"], height=180)
        st.download_button(
            "⬇️ Descargar OCR",
            data=st.session_state["ocr_text"],
            file_name="ocr.txt",
            mime="text/plain",
            use_container_width=True
        )

    if "groq_out" in st.session_state:
        st.markdown("#### Salida de Groq")
        st.write(st.session_state["groq_out"])
        st.download_button(
            "⬇️ Descargar salida Groq",
            data=st.session_state["groq_out"],
            file_name="groq_out.txt",
            mime="text/plain",
            use_container_width=True
        )

st.info("Sugerencia: si el OCR sale pobre, intenta con imágenes más nítidas (300–400 DPI) y revisa los idiomas configurados.")
