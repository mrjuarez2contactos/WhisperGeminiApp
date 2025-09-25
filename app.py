# app.py
# Whisper + Gemini 1.5 con Streamlit
# Versión final estable basada en el código original del usuario.
# Incluye una pausa automática para evitar errores de cuota de la API.
# Modelo de Gemini actualizado a una versión superior.

import os
import re
import subprocess
from pathlib import Path
import tempfile
import zipfile
import time

import streamlit as st
from tenacity import retry, stop_after_attempt, wait_exponential

# =========================
# CONFIG INICIAL DE PÁGINA
# =========================
st.set_page_config(page_title="Whisper + Resúmenes", layout="wide")
st.title("📞 Transcripción y Resumen de Audios")

# =========================
# UTILIDADES DE ARCHIVOS
# =========================
def read_text_file(p: Path, max_chars=None) -> str:
    txt = p.read_text(encoding="utf-8", errors="ignore") if p.exists() else ""
    if max_chars and len(txt) > max_chars:
        txt = txt[:max_chars]
    return txt

def write_text_file(p: Path, content: str):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")

def secs_to_timestamp(secs: float):
    if secs is None: secs = 0.0
    ms = int(round((secs - int(secs)) * 1000))
    h, rem = divmod(int(secs), 3600)
    m, s = divmod(rem, 60)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"

def segments_to_srt(segments):
    lines = []
    for i, s in enumerate(segments, 1):
        text = (s.text or "").strip()
        if not text: continue
        lines.append(str(i))
        lines.append(f"{secs_to_timestamp(s.start)} --> {secs_to_timestamp(s.end)}")
        lines.append(text)
        lines.append("")
    return "\n".join(lines)

# --- Conversión Robusta de AMR a WAV ---
def _get_ffmpeg_path():
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as e:
        st.error("Error crítico: No se encontró 'imageio-ffmpeg'. Asegúrate de que esté en requirements.txt.")
        st.stop()

def convert_to_wav_if_needed(src_path: Path) -> Path:
    if src_path.suffix.lower() == ".wav":
        return src_path
    
    ffmpeg = _get_ffmpeg_path()
    dst = src_path.with_suffix(".wav")
    cmd = [ffmpeg, "-y", "-i", str(src_path), "-ac", "1", "-ar", "16000", str(dst)]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0 or not dst.exists():
        msg = proc.stderr or proc.stdout or "ffmpeg error desconocido"
        raise RuntimeError(f"ffmpeg falló al convertir {src_path.name} -> WAV:\n{msg[:800]}")
    return dst

# =========================
# WHISPER (faster-whisper)
# =========================
@st.cache_resource
def load_whisper_model(model_name: str):
    from faster_whisper import WhisperModel
    device = "cpu"
    compute_type = "int8"
    model = WhisperModel(model_name, device=device, compute_type=compute_type)
    return model, device, compute_type

def transcribe_files(model, input_dir: Path, output_dir: Path, language: str, task: str):
    valid_exts = {".amr", ".mp3", ".mp4", ".wav", ".m4a", ".ogg", ".flac"}
    files_to_process = [p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in valid_exts]
    if not files_to_process:
        st.warning("No se encontraron archivos de audio válidos en el directorio de entrada.")
        return

    progress_placeholder = st.empty()
    for i, f in enumerate(files_to_process, 1):
        progress_text = f"Transcribiendo: {f.name} ({i}/{len(files_to_process)})"
        progress_placeholder.progress(i / len(files_to_process), text=progress_text)
        try:
            audio_path = convert_to_wav_if_needed(f)
            segments_iter, _ = model.transcribe(str(audio_path), language=language, task=task)
            segments = list(segments_iter)
            base_name = f.stem
            write_text_file(output_dir / f"{base_name}.txt", "".join((s.text or "") for s in segments).strip())
            write_text_file(output_dir / f"{base_name}.srt", segments_to_srt(segments))
        except Exception as e:
            st.error(f"Error al transcribir {f.name}: {e}")
    progress_placeholder.empty()

# =========================
# RESÚMENES (Gemini 1.5)
# =========================
def build_summary_prompt(user_context: str):
    return (
        "Eres un asistente que resume llamadas comerciales sobre compra-venta de camarón, pulpo y tilapia.\\n"
        "Objetivo: entregar un resumen útil para decisiones comerciales.\\n"
        "- Conserva cifras, tamaños/tallas y acuerdos (precio, cantidad, fechas, flete, almacenamiento).\\n"
        "- Convierte nombres propios a Mayúscula Inicial. No inventes datos.\\n"
        f"\\n[Contexto del usuario]\\n{user_context}\\n"
        "\\n[Instrucciones de formato]\\n"
        "- Devuelve un único párrafo de 4–8 oraciones.\\n"
        "- Inicia con un renglón 'Contacto: NOMBRE | Fecha: AAAA-MM-DD HH:MM' si está en el nombre del archivo o la transcripción.\\n"
    )

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def summarize_with_gemini(api_key, model_name, user_context, transcript):
    import google.generativeai as genai
    genai.configure(api_key=api_key)
    prompt = build_summary_prompt(user_context)
    model = genai.GenerativeModel(model_name)
    response = model.generate_content([prompt, "\\n[Transcripción]\\n" + transcript])
    return (response.text or "").strip()

# =========================
# SIDEBAR (CONFIG)
# =========================
with st.sidebar:
    st.markdown("### 🎙️ Configuración de Whisper")
    model_name = st.selectbox("Modelo", ["tiny", "base", "small", "medium"], index=2)
    language = st.text_input("Idioma (ej. 'es', 'en')", "es")
    task = st.selectbox("Tarea", ["transcribe", "translate"], index=0)

    st.markdown("### 🔐 Clave de API de Gemini")
    # Se lee la clave desde los "Secrets" de Streamlit para mayor seguridad.
    GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY", "")
    # >>>>> CAMBIO CLAVE: MODELO ACTUALIZADO <<<<<
    GEMINI_MODEL = st.text_input("Modelo Gemini", "gemini-2.5-pro")

    st.markdown("### 🧭 Contexto para Resúmenes")
    USER_CONTEXT = st.text_area(
        "Contexto de negocio",
        "Me dedico a la compra-venta de camarón, pulpo y filete de tilapia...",
        height=150
    )

# Carpeta de salida relativa
OUTPUT_DIR = Path("transcripciones_output")
OUTPUT_DIR.mkdir(exist_ok=True)

model, device, compute_type = load_whisper_model(model_name)
st.sidebar.info(f"Whisper listo en `{device}` ({compute_type})")

# =========================
# UI PRINCIPAL (TABS)
# =========================
tab1, tab2, tab3 = st.tabs(["1) Transcribir Audios", "2) Generar Resúmenes", "3) Descargar Resultados"])

with tab1:
    st.header("Sube tus archivos de audio")
    uploaded_files = st.file_uploader(
        "Selecciona uno o más archivos de audio",
        accept_multiple_files=True
    )

    if st.button("▶️ Iniciar Transcripción", disabled=not uploaded_files):
        with tempfile.TemporaryDirectory() as tmpdir:
            input_dir = Path(tmpdir)
            for uf in uploaded_files or []:
                p = input_dir / uf.name
                p.write_bytes(uf.getbuffer())
            
            with st.spinner("Procesando... esto puede tardar varios minutos."):
                transcribe_files(model, input_dir, OUTPUT_DIR, language, task)
            st.success("¡Transcripción completada!")
            st.balloons()

with tab2:
    st.header("Genera resúmenes de las transcripciones")
    txt_files = sorted(OUTPUT_DIR.glob("*.txt"))
    if not txt_files:
        st.info("Aún no hay transcripciones. Sube y transcribe archivos en la Pestaña 1.")
    else:
        st.write(f"Se encontraron **{len(txt_files)}** transcripciones listas para resumir.")

    if st.button("▶️ Generar Resúmenes", disabled=not txt_files):
        if not GOOGLE_API_KEY:
            st.error("Por favor, introduce tu GOOGLE_API_KEY en los 'Secrets' de la aplicación.")
        else:
            gem_dir = OUTPUT_DIR / "_geminis"
            gem_dir.mkdir(exist_ok=True)
            progress_placeholder = st.empty()
            results_placeholder = st.container()

            for i, f in enumerate(txt_files, 1):
                progress_text = f"Resumiendo: {f.name} ({i}/{len(txt_files)})"
                progress_placeholder.progress(i / len(txt_files), text=progress_text)
                
                gem_sum = ""
                try:
                    transcript = read_text_file(f, max_chars=100000)
                    gem_sum = summarize_with_gemini(GOOGLE_API_KEY, GEMINI_MODEL, USER_CONTEXT, transcript)
                    write_text_file(gem_dir / f.name, gem_sum)
                except Exception as e:
                    st.error(f"Error con Gemini en {f.name}: {e}")

                with results_placeholder.expander(f"📄 Resultados para: {f.name}"):
                    st.text_area("Resumen Gemini", gem_sum or "No generado.", height=200)

                # >>>>> PAUSA AUTOMÁTICA Y FIJA <<<<<
                if i < len(txt_files): # No pausar después del último archivo
                    progress_placeholder.info(f"Pausa de 15 segundos para no saturar la API...")
                    time.sleep(15)

            progress_placeholder.empty()
            st.success("¡Resúmenes generados!")

with tab3:
    st.header("Descarga todos tus archivos")
    st.write("Haz clic para crear un `.zip` con todas las transcripciones y resúmenes.")

    if st.button("📦 Preparar Archivo .ZIP"):
        zip_path = Path("resultados.zip")
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            files_to_zip = list(OUTPUT_DIR.rglob("*"))
            if not files_to_zip:
                st.warning("No hay archivos para comprimir.")
            else:
                for file_path in files_to_zip:
                    if file_path.is_file():
                        zipf.write(file_path, arcname=file_path.relative_to(OUTPUT_DIR))
                with open(zip_path, "rb") as f:
                    st.download_button(
                        label="⬇️ Descargar resultados.zip",
                        data=f,
                        file_name="resultados.zip",
                        mime="application/zip"
                    )

