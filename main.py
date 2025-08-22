import os
from typing import List, Tuple

import streamlit as st
from groq import Groq
from PIL import Image, ImageOps
import easyocr
import numpy as np

# =========================
# Configuración de página
# =========================
st.set_page_config(
    page_title="OCR + Groq · ES/EN/PT",
    page_icon="🧩",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================
# Utilidades
# =========================

@st.cache_resource
def load_easyocr_reader(langs: List[str]):
    """
    Carga el lector de EasyOCR en caché (CPU).
    """
    return easyocr.Reader(langs, gpu=False)

def ensure_groq() -> Groq:
    """
    Devuelve cliente Groq desde barra lateral o variable de entorno.
    """
    key = st.session_state.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")
    if not key:
        st.error("Falta GROQ_API_KEY. Ingresa tu key en la barra lateral o como variable de entorno.")
        st.stop()
    return Groq(api_key=key)

def normalize_image(pil_img: Image.Image) -> Image.Image:
    """
    Preprocesamiento suave: RGB + corregir orientación EXIF.
    """
    img = pil_img.convert("RGB")
    img = ImageOps.exif_transpose(img)
    return img

def easyocr_extract_text(pil_img: Image.Image, reader: easyocr.Reader) -> Tuple[str, list]:
    """
    Devuelve: (texto_concatenado, resultados_crudos)
    """
    arr = np.array(pil_img)
    result = reader.readtext(arr)  # [(bbox, text, conf), ...]
    texto = " ".join([r[1] for r in result]).strip()
    return texto, result

def call_groq_analysis(text: str, mode: str, model: str, temperature: float = 0.2) -> str:
    """
    Analiza/sintetiza el texto OCR con Groq.
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

Responde SOLO con viñetas:"""
    else:
        prompt = f"""Del siguiente texto extraído por OCR:
- Corrige errores evidentes de OCR.
- Si son datos, organízalos en tabla Markdown o lista estructurada.
- Si es una pregunta, respóndela brevemente.
- Si es largo, resume puntos clave.

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

def translate_with_groq(text: str, target_lang_label: str, model: str = "llama-3.1-8b-instant") -> str:
    """
    Traduce el texto al idioma objetivo usando Groq.
    target_lang_label: 'English' | 'Portuguese' | etc.
    """
    if not text:
        return ""
    client = ensure_groq()
    prompt = f"""Traduce el siguiente texto al {target_lang_label} de forma natural.
No agregues explicaciones, solo la traducción.

---
{text}
---"""
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0
    )
    return resp.choices[0].message.content.strip()

# =========================
# Interfaz
# =========================

st.title("🧩 OCR + Groq con traducción ES/EN/PT")
st.caption("Sube una imagen → OCR con EasyOCR → Análisis con Groq → Traducciones (Inglés y Portugués).")

with st.sidebar:
    st.header("🔑 Configuración")
    api_key = st.text_input("GROQ_API_KEY", type="password", help="También puedes definirla como variable de entorno.")
    if api_key:
        st.session_state["GROQ_API_KEY"] = api_key

    st.subheader("🛠️ Opciones OCR/LLM")
    langs_str = st.text_input("Idiomas OCR (EasyOCR)", value="es,en", help="Separados por coma, ej: es,en")
    langs = [s.strip() for s in langs_str.split(",") if s.strip()]

    mode = st.radio("Modo de análisis", ["Resumen", "Analizar / Organizar"], index=0)
    model = st.selectbox("Modelo Groq", ["llama-3.1-8b-instant", "llama-3.1-70b-versatile"], index=0)
    temperature = st.slider("Temperature (Groq)", 0.0, 1.0, 0.2, 0.05)

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("🖼️ Sube tu imagen")
    up = st.file_uploader("PNG/JPG/JPEG", type=["png", "jpg", "jpeg"], label_visibility="collapsed")

    if up:
        img = Image.open(up)
        st.image(img, caption="Imagen cargada", use_column_width=True)

        if st.button("✨ Ejecutar OCR + Análisis + Traducciones", type="primary", use_container_width=True):
            # 1) Cargar OCR (cacheado)
            try:
                reader = load_easyocr_reader(langs if langs else ["es", "en"])
            except Exception as e:
                st.error(f"No se pudo cargar EasyOCR: {e}")
                st.stop()

            # 2) OCR
            with st.spinner("🔍 Extrayendo texto (EasyOCR)…"):
                img_norm = normalize_image(img)
                text_es, raw = easyocr_extract_text(img_norm, reader)
                st.session_state["ocr_es"] = text_es
                st.session_state["ocr_raw"] = raw

            # 3) Análisis Groq
            with st.spinner("🧠 Analizando con Groq…"):
                analysis = call_groq_analysis(text_es, mode=mode, model=model, temperature=temperature)
                st.session_state["groq_out"] = analysis

            # 4) Traducciones
            with st.spinner("🌍 Traduciendo a EN/PT…"):
                st.session_state["ocr_en"] = translate_with_groq(text_es, "English", model=model)
                st.session_state["ocr_pt"] = translate_with_groq(text_es, "Portuguese", model=model)

with col2:
    st.subheader("💡 Resultados")

    # Texto original en ES (OCR)
    if "ocr_es" in st.session_state:
        st.markdown("#### Texto en Español (OCR)")
        st.text_area("Español", st.session_state["ocr_es"], height=160)
        st.download_button(
            "⬇️ Descargar OCR (ES)",
            data=st.session_state["ocr_es"],
            file_name="ocr_es.txt",
            mime="text/plain",
            use_container_width=True
        )

    # Traducciones
    if "ocr_en" in st.session_state:
        st.markdown("#### Traducción al Inglés")
        st.text_area("English", st.session_state["ocr_en"], height=160)
        st.download_button(
            "⬇️ Descargar EN",
            data=st.session_state["ocr_en"],
            file_name="ocr_en.txt",
            mime="text/plain",
            use_container_width=True
        )

    if "ocr_pt" in st.session_state:
        st.markdown("#### Traducción al Portugués")
        st.text_area("Português", st.session_state["ocr_pt"], height=160)
        st.download_button(
            "⬇️ Descargar PT",
            data=st.session_state["ocr_pt"],
            file_name="ocr_pt.txt",
            mime="text/plain",
            use_container_width=True
        )

    # Análisis / Resumen Groq
    if "groq_out" in st.session_state:
        st.markdown("#### Análisis / Resumen (Groq)")
        st.write(st.session_state["groq_out"])
        st.download_button(
            "⬇️ Descargar análisis",
            data=st.session_state["groq_out"],
            file_name="analisis_groq.txt",
            mime="text/plain",
            use_container_width=True
        )

st.info("Consejo: si el OCR no se ve bien, usa imágenes más nítidas (300–400 DPI) y ajusta los idiomas en la barra lateral.")
