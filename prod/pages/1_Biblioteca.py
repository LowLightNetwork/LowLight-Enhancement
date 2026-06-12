"""Página Biblioteca: galería pública de imágenes mejoradas compartidas.

Streamlit detecta automáticamente la carpeta `pages/` y agrega esta página
a la navegación lateral de la app.
"""

from datetime import datetime

import streamlit as st

from library_db import fetch_library, supabase_configured


def _escape_markdown(text: str) -> str:
    return "".join(f"\\{char}" if char in r"\`*_{}[]<>()#+-.!|" else char for char in text)

st.set_page_config(
    page_title="Biblioteca — Low-Light Enhancement",
    page_icon="🖼️",
    layout="wide",
)

st.title("Biblioteca")
st.markdown(
    "Imágenes mejoradas con la app que los usuarios decidieron compartir, "
    "con la original al lado para ver la comparación. Para sumar la tuya, "
    "mejorá una foto en la página principal y usá **Subir a la biblioteca**."
)

if not supabase_configured():
    st.info(
        "La biblioteca no está disponible: faltan las credenciales de "
        "Supabase (`SUPABASE_URL` y `SUPABASE_KEY` en los secrets)."
    )
    st.stop()

try:
    entries = fetch_library()
except Exception as e:
    st.error(f"No se pudo cargar la biblioteca: {e}")
    st.stop()

if not entries:
    st.info("La biblioteca está vacía por ahora. ¡Sé el primero en compartir una imagen!")
    st.stop()

st.caption(f"{len(entries)} imagen(es) compartida(s)")

for entry in entries:
    try:
        fecha = datetime.fromisoformat(entry["created_at"]).strftime("%d/%m/%Y %H:%M")
    except ValueError:
        fecha = entry["created_at"]
    st.markdown(f"**{_escape_markdown(str(entry['usertag']))}** · {fecha}")

    col_orig, col_enh = st.columns(2)
    with col_orig:
        st.caption("Original")
        if entry.get("original_url"):
            st.image(entry["original_url"], use_container_width=True)
        else:
            st.info("Esta entrada no tiene la imagen original guardada.")
    with col_enh:
        st.caption("Mejorada")
        st.image(entry["image_url"], use_container_width=True)

    st.divider()
