# app.py
# Whisper + Gemini 2.5 (resúmenes) con Streamlit
# Versión final estable con control manual, soporte AMR robusto y limpieza.

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
st.title("📞 Transcripción y Resumen de Audios (Control Manual)")

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
    proc = subprocess.run(cmd, capture_output=True, text=True)
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
# RESÚMENES (Gemini 2.5)
# =========================
def build_summary_prompt(user_context: str) -> str:
    return (
        "Eres un asistente experto que resume llamadas telefónicas del año 2025 sobre la comercialización de mariscos. Tu resumen debe ser un único párrafo conciso y directo.\\n\\n"
        "ENFOQUE PRINCIPAL: Extrae y resume ÚNICAMENTE los detalles comerciales clave.\\n\\n"
        "REGLAS DE FORMATO ESTRICTAS:\\n"
        "1. REGLA CRÍTICA: Escribe TODOS los nombres propios de personas con Mayúscula Inicial (Ej: 'Juan Pérez').\\n"
        "2. OMITE por completo cualquier mención a 'Orador 1' o los nombres de los participantes.\\n"
        "3. El resultado final debe ser un solo párrafo.\\n"
        f"\\n[CONTEXTO DEL NEGOCIO]\\n{user_context.strip()}\\n"
        "\\n[INSTRUCCIONES FINALES]\\nA continuación se te proporcionará la transcripción."
    )

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def summarize_with_gemini(api_key: str, model_name: str, user_context: str, transcript: str) -> str:
    if not api_key:
        raise RuntimeError("Falta GOOGLE_API_KEY en los Secrets de la aplicación.")
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
    st.markdown("### 🎙️ Whisper")
    model_name = st.selectbox("Modelo", ["tiny", "base", "small", "medium"], index=2)
    language = st.text_input("Idioma (ej. 'es', 'en')", "es")
    task = st.selectbox("Tarea", ["transcribe", "translate"], index=0)

    st.markdown("### 🔐 Gemini 1.5")
    GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY", "")
    GEMINI_MODEL = st.text_input("Modelo Gemini", "gemini-1.5-flash")

    st.markdown("### 🧭 Contexto para Resúmenes")
    USER_CONTEXT = st.text_area(
        "Contexto de negocio",
        value=(
            "Me dedico a la compra venta de camarón, pulpo y pescado.\\n\\n"
            "**CAMARÓN VIVO:** Talla por gramos (8-45g). Precio/kg = gramos + 100.\\n"
            "**CAMARÓN CONGELADO:** Tallas 16/20, 21/25, etc. (unidades/libra).\\n"
            "**PULPO:** Tallas 1/2, 2/4 (unidades/libra).\\n"
            "**TILAPIA:** Tallas 3/5, 5/7 (onzas/filete) con % de agua."
        ),
        height=200
    )

# La carpeta de salida ahora es temporal dentro de la sesión de Streamlit
OUTPUT_DIR = Path("transcripciones_output")
OUTPUT_DIR.mkdir(exist_ok=True)

model, device, compute_type = load_whisper_model(model_name)
st.sidebar.info(f"Whisper listo en `{device}` ({compute_type})")

# =========================
# UI PRINCIPAL (TABS)
# =========================
tab1, tab2, tab3 = st.tabs([
    "1) Transcribir Audios",
    "2) Generar Resúmenes",
    "3) Descargar / Limpiar"
])

# ---------- TAB 1: Transcribir ----------
with tab1:
    st.header("Sube tus archivos de audio")
    uploaded_files = st.file_uploader(
        "Selecciona uno o más archivos de audio",
        type=['amr', 'mp3', 'mp4', 'wav', 'm4a', 'ogg', 'flac'],
        accept_multiple_files=True
    )

    if st.button("▶️ Iniciar Transcripción", disabled=not uploaded_files):
        with tempfile.TemporaryDirectory() as tmpdir:
            input_dir = Path(tmpdir)
            for f in uploaded_files:
                (input_dir / f.name).write_bytes(f.getbuffer())
            with st.spinner("Procesando..."):
                transcribe_files(model, input_dir, OUTPUT_DIR, language, task)
        st.success("¡Transcripción completada!")

# ---------- TAB 2: Resumir (solo nuevos) ----------
with tab2:
    st.header("Genera resúmenes (Control Manual Estable)")
    gem_dir = OUTPUT_DIR / "_geminis"
    gem_dir.mkdir(exist_ok=True)

    if 'summary_idx' not in st.session_state: st.session_state.summary_idx = 0
    if 'summaries' not in st.session_state: st.session_state.summaries = {}
    if 'error_info' not in st.session_state: st.session_state.error_info = None
        
    all_transcripts = sorted(OUTPUT_DIR.glob("*.txt"))
    existing_summary_stems = {p.stem for p in gem_dir.glob("*.txt")}
    to_summarize = [p for p in all_transcripts if p.stem not in existing_summary_stems]
    
    st.write(f"Transcripciones listas para resumir: **{len(to_summarize)}**")
    
    if st.button("Resetear Proceso"):
        st.session_state.summary_idx = 0
        st.session_state.summaries = {}
        st.session_state.error_info = None

    st.markdown("---")

    idx = st.session_state.summary_idx
    
    if st.session_state.error_info and idx < len(to_summarize):
        failed_file_name = to_summarize[idx].name
        st.error(f"Pausa por error de cuota en: **{failed_file_name}**")
        st.warning("⏳ Por favor, espere un minuto para que la cuota de la API se recupere.")
        
        if st.button("He esperado. Reintentar Archivo Anterior"):
            st.session_state.error_info = None

    elif idx < len(to_summarize):
        next_file = to_summarize[idx]
        st.info(f"Siguiente archivo a procesar ({idx + 1}/{len(to_summarize)}): **{next_file.name}**")

        if st.button("▶️ Resumir Siguiente Archivo"):
            if not GOOGLE_API_KEY:
                st.warning("Falta la GOOGLE_API_KEY en los Secrets de la aplicación.")
            else:
                with st.spinner(f"Resumiendo {next_file.name}..."):
                    try:
                        transcript = read_text_file(next_file, max_chars=120_000)
                        gem_sum = summarize_with_gemini(GOOGLE_API_KEY, GEMINI_MODEL, USER_CONTEXT, transcript)
                        st.session_state.summaries[next_file.name] = gem_sum
                        (gem_dir / (next_file.stem + ".txt")).write_text(gem_sum, encoding="utf-8")
                        st.session_state.summary_idx += 1
                    except Exception as e:
                        if "ResourceExhausted" in str(e) or "429" in str(e):
                            st.session_state.error_info = str(e)
                        else:
                            st.error(f"Error inesperado con {next_file.name}: {e}")
                            st.session_state.summaries[next_file.name] = f"ERROR: {e}"
                            st.session_state.summary_idx += 1
    
    else:
        if to_summarize: st.success("🎉 ¡Todos los archivos han sido procesados! 🎉")

    if st.session_state.summaries:
        st.markdown("---")
        st.subheader("Resúmenes Generados en esta Sesión")
        for fname, summary in reversed(list(st.session_state.summaries.items())):
            with st.expander(f"📄 {fname}"):
                st.text_area("Resumen", summary or "No generado.", height=220, key=f"summary_{fname}")

# ---------- TAB 3: ZIP + Limpieza ----------
with tab3:
    st.header("Descargar resultados y limpiar")
    if st.button("📦 Preparar ZIP para Descargar"):
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
                    st.download_button("⬇️ Descargar resultados.zip", f, "resultados.zip", "application/zip")
                
                st.success("ZIP listo.")

    st.divider()
    st.subheader("🧹 Limpieza Manual")
    if st.button("🗑️ Borrar TODOS los archivos (Transcripciones y Resúmenes)"):
        count = 0
        for p in OUTPUT_DIR.rglob("*"):
            if p.is_file():
                try:
                    p.unlink()
                    count += 1
                except:
                    pass
        st.success(f"Se eliminaron {count} archivos. La aplicación está limpia.")
        st.session_state.summary_idx = 0
        st.session_state.summaries = {}
        st.session_state.error_info = None

