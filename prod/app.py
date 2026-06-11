"""App Streamlit de Low-Light Image Enhancement con Zero-DCE-FT++.

El usuario sube una foto oscura y el modelo devuelve una versión más clara.
Interfaz separada de la lógica auxiliar (utils.py) según la consigna.
"""

import io
import time

import streamlit as st
from PIL import Image

from library_db import supabase_configured, upload_to_library
from utils import enhance_image, load_model

st.set_page_config(
    page_title="Low-Light Enhancement — Zero-DCE-FT++",
    page_icon="🌙",
    layout="wide",
)


@st.cache_resource
def get_model():
    """Carga el modelo una sola vez (cacheado entre interacciones)."""
    return load_model()


model, metadata = get_model()

# ---------------------------------------------------------------------------
# Encabezado
# ---------------------------------------------------------------------------

st.title("Low-Light Image Enhancement")
st.markdown(
    "Subí una foto **oscura** y el modelo **Zero-DCE-FT++** devuelve una versión "
    "más clara. Es una red liviana (~109k parámetros) que estima curvas de "
    "iluminación (DCE-Net) y elimina artefactos con un mini-denoiser (DnCNN), "
    "fine-tuneada sobre el dataset LOL (v1 + v2-real)."
)

with st.expander("Detalles del modelo"):
    st.markdown(
        f"""
- **Arquitectura:** `{metadata.get("arch", "ZeroDCE_PP")}`
- **Métricas en test (con TTA):** PSNR = **{metadata.get("test_psnr", "?")} dB** ·
  SSIM = **{metadata.get("test_ssim", "?")}** · LPIPS = **{metadata.get("test_lpips", "?")}**
- **Preprocesamiento:** {metadata.get("preprocess_note", "RGB, dividir por 255")} (idéntico al usado en test)
- **Pesos:** EMA (decay = {metadata.get("ema_decay", 0.999)})
- **Repositorio:** [GitHub](https://github.com/jeanpaulmst/LowLight-Enhancement)
"""
    )

# ---------------------------------------------------------------------------
# Controles
# ---------------------------------------------------------------------------

use_tta = st.checkbox(
    "Usar TTA — Test-Time Augmentation (mejor calidad, mucho más lento en CPU)",
    value=False,
    help=(
        "Promedia las predicciones sobre las 8 simetrías de la imagen "
        "(grupo diédrico D4). Es la configuración con la que el modelo "
        "obtuvo su mejor PSNR en test, pero en CPU multiplica el tiempo "
        "de inferencia ~8×, por lo que la imagen se procesa a resolución "
        "reducida (máx. 600 px de lado)."
    ),
)

uploaded = st.file_uploader(
    "Subí una imagen oscura (JPG o PNG)",
    type=["jpg", "jpeg", "png"],
)

# ---------------------------------------------------------------------------
# Inferencia y resultados
# ---------------------------------------------------------------------------

if uploaded is not None:
    original = Image.open(uploaded)

    # Cachear el resultado en session_state para que interacciones posteriores
    # (ej. subir a la biblioteca) no vuelvan a correr la inferencia.
    result_key = (uploaded.file_id, use_tta)
    if st.session_state.get("result_key") != result_key:
        with st.spinner("Mejorando la imagen..."):
            t0 = time.time()
            st.session_state["enhanced"] = enhance_image(
                model, original, use_tta=use_tta
            )
            st.session_state["elapsed"] = time.time() - t0
            st.session_state["result_key"] = result_key
            st.session_state["shared"] = False

    enhanced = st.session_state["enhanced"]
    elapsed = st.session_state["elapsed"]

    col_in, col_out = st.columns(2)
    with col_in:
        st.subheader("Original")
        st.image(original, use_container_width=True)
    with col_out:
        st.subheader("Mejorada")
        st.image(enhanced, use_container_width=True)

    st.caption(
        f"Inferencia en {elapsed:.1f} s "
        f"({'con' if use_tta else 'sin'} TTA, CPU) · "
        f"resolución procesada: {enhanced.width}×{enhanced.height}"
    )

    buf = io.BytesIO()
    enhanced.save(buf, format="PNG")
    st.download_button(
        "Descargar imagen mejorada (PNG)",
        data=buf.getvalue(),
        file_name="imagen_mejorada.png",
        mime="image/png",
    )

    # -----------------------------------------------------------------------
    # Compartir en la biblioteca (opt-in: nada se sube por defecto)
    # -----------------------------------------------------------------------

    st.divider()
    st.subheader("Compartir en la biblioteca")

    if not supabase_configured():
        st.warning(
            "La biblioteca no está disponible: faltan las credenciales de "
            "Supabase (`SUPABASE_URL` y `SUPABASE_KEY` en los secrets)."
        )
    elif st.session_state.get("shared"):
        st.success("¡Tu imagen ya está en la biblioteca! Podés verla en la página **Biblioteca**.")
    else:
        st.markdown(
            "Si querés, podés subir tu resultado a la biblioteca pública de la "
            "app para que otros lo vean. Solo se comparte la imagen mejorada."
        )
        usertag = st.text_input(
            "Tu nombre o usertag",
            max_chars=50,
            placeholder="ej. @nico",
        )
        if st.button("Subir a la biblioteca"):
            if not usertag.strip():
                st.error("Ingresá un nombre o usertag antes de subir.")
            else:
                try:
                    with st.spinner("Subiendo a la biblioteca..."):
                        upload_to_library(enhanced, usertag)
                    st.session_state["shared"] = True
                    st.rerun()
                except Exception as e:
                    st.error(f"No se pudo subir la imagen: {e}")
else:
    st.info(
        "Esperando una imagen... Funciona mejor con fotos tomadas con poca luz. "
        "Si subís una imagen ya bien iluminada, el modelo puede sobreexponerla."
    )
