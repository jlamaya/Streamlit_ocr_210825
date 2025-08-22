import io
import os
import time
from typing import List, Tuple

import streamlit as st
from PIL import Image
import pytesseract
import fitz  # PyMuPDF
from groq import Groq

# ---------------------------
# Configuración básica
# ---------------------------
st.set_page_config(page_title="OCR + Groq", page_icon="🧠", layout="wide")

DEFAULT_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-70b-versatile")
DEFAULT_OCR_LANG = os.getenv("OCR_LANG", "spa")  # 'spa', 'eng', 'spa+eng'


def ensure_groq() -> Groq:
    api_key = os.getenv("GROQ_API_KEY") or st.session_state.get("GROQ_API_KEY")
    if not api_key:
        st.error("⚠️ No se encontró GROQ_API_KEY. Ponla en la barra lateral.")
        st.stop()
    return Groq(api_key=api_key)


def ocr_image(pil_img: Image.Image, lang: str) -> str:
    gray = pil_img.convert("L")  # escala de grises
    text = pytesseract.image_to_string(gray, lang=lang)
    return text


def pdf_to_images(pdf_bytes: bytes, dpi: int = 200) -> List[Image.Image]:
    images = []
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        for page in doc:
            pix = page.get_pixmap(dpi=dpi, alpha=False)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            images.append(img)
    return images


def extract_text(file_bytes: bytes, filename: str, lang: str, dpi_pdf: int = 200) -> Tuple[str, List[str]]:
    name = filename.lower()
    per_page = []
    if name.endswith(".pdf"):
        imgs = pdf_to_images(file_bytes, dpi=dpi_pdf)
        for img in imgs:
            per_page.append(ocr_image(img, lang=lang))
        return "\n\n".join(per_page), per_page
    else:
        pil = Image.open(io.BytesIO(file_bytes))
        txt = ocr_image(pil, lang=lang)
        return txt, [txt]


def call_groq(task: str, text: str, model: str, temperature: float = 0.2, extra_instructions: str = "") -> str:
    client = ensure_groq()

    if task == "Resumen breve":
        user_prompt = (
            "Resume de forma concisa (bullets) el siguiente texto OCR. "
            "Ignora ruido típico de OCR. Devuelve 5-8 viñetas claras.\n\n"
            f"=== TEXTO OCR ===\n{text}\n"
        )
        system = "Eres un asistente que resume con precisión y sin inventar datos."
    elif task == "Extracción de campos (JSON)":
        user_prompt = (
            "Del texto OCR, extrae los campos en JSON con este esquema: "
            "{'titulo': str|nullable, 'fecha': str|nullable, 'entidades': [str], 'monto': str|nullable, 'otros_campos': dict}. "
            "No agregues comentarios, responde SOLO el JSON válido.\n\n"
            f"=== TEXTO OCR ===\n{text}\n"
        )
        system = "Eres un extractor de información estricto. Solo devuelves JSON válido."
    else:  # Chat libre
        user_prompt = f"{extra_instructions}\n\n=== TEXTO OCR ===\n{text}\n"
        system = "Eres un asistente útil que razona paso a paso."

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
    )
    return resp.choices[0].message.content.strip()


# ---------------------------
# UI
# ---------------------------
st.title("🧠 OCR + Groq")
st.caption("Sube un PDF o imagen, aplica OCR y procesa el texto con Groq.")

with st.sidebar:
    st.header("⚙️ Configuración")
    groq_key_in = st.text_input("GROQ_API_KEY", type="password")
    if groq_key_in:
        st.session_state["GROQ_API_KEY"] = groq_key_in

    model = st.selectbox("Modelo Groq", ["llama-3.1-70b-versatile", "llama-3.1-8b-instant"], index=0)
    ocr_lang = st.text_input("Idioma OCR", value=DEFAULT_OCR_LANG, help="Ej: spa, eng, spa+eng")
    dpi_pdf = st.slider("DPI PDF", min_value=120, max_value=300, value=200, step=20)
    temperature = st.slider("Temperature (Groq)", 0.0, 1.0, 0.2, 0.05)
    task = st.radio("Tarea", ["Resumen breve", "Extracción de campos (JSON)", "Chat libre"], index=0)
    extra = st.text_area("Instrucciones extra (para Chat libre)")

uploaded = st.file_uploader("📂 Sube PDF o imagen", type=["pdf", "png", "jpg", "jpeg", "tiff"])

if uploaded:
    st.write(f"Archivo: **{uploaded.name}**")
    t0 = time.time()
    text_total, textos_pag = extract_text(uploaded.read(), uploaded.name, lang=ocr_lang, dpi_pdf=dpi_pdf)
    st.success(f"OCR listo en {time.time()-t0:.2f} s")

    with st.expander("Texto OCR completo"):
        st.text_area("Texto OCR", text_total, height=250)

    if st.button("🚀 Ejecutar en Groq"):
        t1 = time.time()
        result = call_groq(task=task, text=text_total, model=model, temperature=temperature, extra_instructions=extra)
        st.info(f"Groq respondió en {time.time()-t1:.2f} s")
        if task == "Extracción de campos (JSON)":
            st.code(result, language="json")
        else:
            st.write(result)
