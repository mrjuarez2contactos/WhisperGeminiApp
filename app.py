# app.py
# Whisper + Gemini 2.5 (resúmenes) con Streamlit
# Lógica de resumen manual paso a paso para máxima estabilidad.

import os
import re
from pathlib import Path
import tempfile
import zipfile
import time

import streamlit as st

# =========================
# CONFIG INICIAL DE PÁGINA
# =========================
st.set_page_config(page_title="Whisper + Resúmenes", layout="wide")
st.title("📞 Transcripción y Resumen de Audios (Whisper + Gemini 2.5)")

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

# =========================
# WHISPER (faster-whisper)
# =========================
@st.cache_resource
def load_whisper_model(model_name: str):
    from faster_whisper import WhisperModel
    # For Streamlit Cloud, force CPU. GPU is not available on the free tier.
    device = "cpu"
    compute_type = "int8"
    model = WhisperModel(model_name, device=device, compute_type=compute_type)
    return model, device, compute_type

def transcribe_files(model, input_dir: Path, output_dir: Path, language: str, task: str):
    valid_exts = {".amr", ".mp3", ".mp4", ".wav", ".m4a", ".ogg", ".flac", ".mpeg4", ".webm", ".wma", ".aac"}
    files_to_process = [p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in valid_exts]
    if not files_to_process:
        st.warning("No se encontraron archivos de audio válidos en el directorio de entrada.")
        return

    progress_placeholder = st.empty()
    for i, f in enumerate(files_to_process, 1):
        progress_text = f"Transcribiendo: {f.name} ({i}/{len(files_to_process)})"
        progress_placeholder.progress(i / len(files_to_process), text=progress_text)
        try:
            segments, _ = model.transcribe(str(f), language=language, task=task)
            segments = list(segments)
            out_txt = output_dir / f"{f.stem}.txt"
            write_text_file(out_txt, "".join((s.text or "") for s in segments).strip())
        except Exception as e:
            st.error(f"Error al transcribir {f.name}: {e}")
    progress_placeholder.empty()

# =========================
# RESÚMENES (Gemini 2.5)
# =========================
def build_summary_prompt(user_context: str) -> str:
    return (
        "Eres un asistente experto que resume llamadas telefónicas del año 2025 sobre la comercialización de mariscos. Tu resumen debe ser un único párrafo conciso y directo.\\n\\n"
        "ENFOQUE PRINCIPAL: Extrae y resume ÚNICAMENTE los detalles comerciales clave. Ignora por completo conversaciones banales, chistes, saludos o cualquier tema no relacionado con el negocio.\\n\\n"
        "TEMAS A INCLUIR (si se mencionan):\\n"
        "- Acuerdos de precio, cantidad y fechas de entrega.\\n"
        "- Detalles sobre fletes y transporte.\\n"
        "- Menciones sobre almacenamiento en congeladoras.\\n"
        "- Especificaciones de producto (tallas, calidad, porcentajes de agua, etc.).\\n\\n"
        "REGLAS DE FORMATO ESTRICTAS:\\n"
        "1. REGLA CRÍTICA: Escribe TODOS los nombres propios de personas con Mayúscula Inicial (Ej: 'Juan Pérez', 'María'). Es muy importante.\\n"
        "2. OMITE por completo cualquier mención a 'Orador 1' o los nombres de los participantes. El resumen debe ser impersonal y directo al grano.\\n"
        "3. El resultado final debe ser un solo párrafo, ideal para copiar y pegar en una celda de una hoja de cálculo.\\n"
        f"\\n[CONTEXTO DEL NEGOCIO PROPORCIONADO POR EL USUARIO]\\n{user_context.strip()}\\n"
        "\\n[INSTRUCCIONES FINALES]\\nA continuación se te proporcionará la transcripción. Procesa el texto siguiendo TODAS las reglas anteriores para generar el resumen."
    )

def summarize_with_gemini(api_key: str, model_name: str, user_context: str, transcript: str) -> str:
    if not api_key:
        raise RuntimeError("Falta GOOGLE_API_KEY en la barra lateral.")
    import google.generativeai as genai
    genai.configure(api_key=api_key)
    prompt = build_summary_prompt(user_context)
    model = genai.GenerativeModel(model_name or "gemini-2.5-pro")
    resp = model.generate_content([prompt, "\\n[Transcripción]\\n" + (transcript or "")])
    return (getattr(resp, "text", "") or "").strip()

# =========================
# SIDEBAR (CONFIG)
# =========================
with st.sidebar:
    st.markdown("### 🎙️ Whisper")
    model_name = st.selectbox("Modelo", ["tiny", "base", "small", "medium"], index=2)
    language = st.text_input("Idioma (ej. 'es', 'en')", "es")
    task = st.selectbox("Tarea", ["transcribe", "translate"], index=0)

    st.markdown("### 🔐 Gemini 2.5")
    # En Streamlit Cloud, las claves se guardan de forma segura en los "Secrets"
    GOOGLE_API_KEY = st.text_input("GOOGLE_API_KEY", type="password", value=st.secrets.get("GOOGLE_API_KEY", ""))
    GEMINI_MODEL = st.text_input("Modelo Gemini", value="gemini-2.5-pro")

    st.markdown("### 🧭 Contexto para Resúmenes")
    USER_CONTEXT = st.text_area(
        "Contexto de negocio",
        value=(
            "Me dedico a la compra venta de camarón, pulpo y pescado.\\n\\n"
            "**CAMARÓN EN PIE DE BORDO (VIVO):**\\n"
            "La talla se maneja por gramos (de 8g a 45g, con un mínimo de 5g). El precio por kilo se calcula como: (gramos de la talla) + 100. Ejemplo: Un camarón de 14g cuesta 14 + 100 = $114/kg.\\n\\n"
            "**CAMARÓN CONGELADO (SIN CABEZA):**\\n"
            "Se clasifica en tallas estándar (16/20, 21/25, ..., 91/110). Estas tallas indican el número de camarones por libra (454g). Para estimar el peso original con cabeza, se calcula: (peso sin cabeza) / 0.70.\\n\\n"
            "**PULPO:**\\n"
            "Se maneja por tallas 1/2 y 2/4 (número de pulpos por libra).\\n\\n"
            "**FILETE DE TILAPIA:**\\n"
            "Se maneja por tallas 3/5 y 5/7 (onzas por filete). Se especifica con un porcentaje de agua."
        ),
        height=350
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
        type=['amr', 'mp3', 'mp4', 'wav', 'm4a', 'ogg', 'flac', 'mpeg4', 'webm', 'wma', 'aac'],
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
        st.balloons()

# ---------- TAB 2: Resumir (solo nuevos) ----------
with tab2:
    st.header("Genera resúmenes de las transcripciones (Control Manual)")
    gem_dir = OUTPUT_DIR / "_geminis"
    gem_dir.mkdir(exist_ok=True)

    # Inicialización del estado de la sesión
    if 'summary_idx' not in st.session_state: st.session_state.summary_idx = 0
    if 'summaries' not in st.session_state: st.session_state.summaries = {}
    if 'error_info' not in st.session_state: st.session_state.error_info = None
        
    # Determinar la lista de archivos que faltan por resumir
    all_transcripts = sorted([p for p in OUTPUT_DIR.glob("*.txt") if not p.name.endswith(".gem25.txt")])
    existing_summary_stems = {p.stem.replace('.gem25', '') for p in gem_dir.glob("*.gem25.txt")}
    to_summarize = [p for p in all_transcripts if p.stem not in existing_summary_stems]
    
    st.write(f"Transcripciones listas para resumir: **{len(to_summarize)}**")
    
    # Botón para reiniciar el proceso
    if st.button("Resetear Proceso"):
        st.session_state.summary_idx = 0
        st.session_state.summaries = {}
        st.session_state.error_info = None
        st.experimental_rerun() # Refrescar la UI después de resetear

    st.markdown("---")

    idx = st.session_state.summary_idx
    
    # Si estamos en modo de pausa por un error
    if st.session_state.error_info and idx < len(to_summarize):
        failed_file_name = to_summarize[idx].name
        st.error(f"Pausa por error de cuota en el archivo: **{failed_file_name}**")
        st.warning("⏳ Por favor, espere un minuto para que la cuota de la API se recupere.")
        
        if st.button("He esperado. Reintentar Archivo Anterior"):
            st.session_state.error_info = None
            st.experimental_rerun()

    # Si no estamos en pausa y aún quedan archivos
    elif idx < len(to_summarize):
        next_file = to_summarize[idx]
        st.info(f"Siguiente archivo a procesar ({idx + 1}/{len(to_summarize)}): **{next_file.name}**")

        if st.button("▶️ Resumir Siguiente Archivo"):
            with st.spinner(f"Resumiendo {next_file.name}..."):
                try:
                    transcript = read_text_file(next_file, max_chars=120_000)
                    gem_sum = summarize_with_gemini(GOOGLE_API_KEY, GEMINI_MODEL, USER_CONTEXT, transcript)
                    st.session_state.summaries[next_file.name] = gem_sum
                    (gem_dir / (next_file.stem + ".gem25.txt")).write_text(gem_sum, encoding="utf-8")
                    st.session_state.summary_idx += 1
                except Exception as e:
                    if "ResourceExhausted" in str(e) or "429" in str(e):
                        st.session_state.error_info = str(e)
                    else:
                        st.error(f"Error inesperado con {next_file.name}: {e}")
                        st.session_state.summaries[next_file.name] = f"ERROR: {e}"
                        st.session_state.summary_idx += 1
            st.experimental_rerun()
    
    # Si ya no quedan archivos
    else:
        if to_summarize: st.success("🎉 ¡Todos los archivos han sido procesados! 🎉")

    # Mostrar siempre los resúmenes generados
    if st.session_state.summaries:
        st.markdown("---")
        st.subheader("Resúmenes Generados en esta Sesión")
        for fname, summary in reversed(list(st.session_state.summaries.items())):
            with st.expander(f"📄 {fname}"):
                st.text_area("Resumen", summary or "No generado.", height=220, key=f"summary_{fname}")

# ---------- TAB 3: ZIP + Limpieza ----------
with tab3:
    st.header("Descargar resultados y limpiar")
    zip_choice = st.radio("¿Qué quieres comprimir?", ["Solo RESÚMENES (_geminis)", "Solo TRANSCRIPCIONES (.txt)", "Todo (transcripciones + resúmenes)"], index=0)
    cleanup_after_zip = st.checkbox("Borrar archivos originales después de crear el ZIP (recomendado)", value=True)
    if st.button("📦 Preparar ZIP"):
        zip_path = Path("resultados.zip")
        files_to_zip, gem_dir = [], OUTPUT_DIR / "_geminis"
        if zip_choice == "Solo RESÚMENES (_geminis)": files_to_zip = list(gem_dir.glob("*.gem25.txt"))
        elif zip_choice == "Solo TRANSCRIPCIONES (.txt)": files_to_zip = [p for p in OUTPUT_DIR.glob("*.txt") if not p.name.endswith(".gem25.txt")]
        else:
            files_to_zip = [p for p in OUTPUT_DIR.glob("*.txt") if not p.name.endswith(".gem25.txt")]
            files_to_zip += list(gem_dir.glob("*.gem25.txt"))
        if not files_to_zip:
            st.warning("No hay archivos para comprimir aún.")
        else:
            with zipfile.ZipFile(zip_path, 'w') as zipf:
                for fp in files_to_zip:
                    if fp.is_file():
                        try: arc = fp.relative_to(OUTPUT_DIR)
                        except Exception: arc = fp.name
                        zipf.write(fp, arcname=str(arc))
            with open(zip_path, "rb") as f:
                st.download_button("⬇️ Descargar resultados.zip", f, "resultados.zip", "application/zip")
            st.success(f"ZIP preparado con {len(files_to_zip)} archivo(s).")
            if cleanup_after_zip:
                st.info("Limpiando archivos que ya fueron comprimidos...")
                count = 0
                for fp in files_to_zip:
                    try: fp.unlink(missing_ok=True); count += 1
                    except Exception: pass
                st.success(f"Limpieza completada. Se eliminaron {count} archivos originales.")

