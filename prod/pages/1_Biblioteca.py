"""Página Biblioteca: galería pública de imágenes mejoradas compartidas.

Streamlit detecta automáticamente la carpeta `pages/` y agrega esta página
a la navegación lateral de la app.
"""

from datetime import datetime

import streamlit as st

from library_db import fetch_library, supabase_configured

st.set_page_config(
    page_title="Biblioteca — Low-Light Enhancement",
    page_icon="🖼️",
    layout="wide",
)

st.title("Biblioteca")
st.markdown(
    "Imágenes mejoradas con la app que los usuarios decidieron compartir. "
    "Para sumar la tuya, mejorá una foto en la página principal y usá "
    "**Subir a la biblioteca**."
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

N_COLS = 3
cols = st.columns(N_COLS)
for i, entry in enumerate(entries):
    with cols[i % N_COLS]:
        st.image(entry["image_url"], use_container_width=True)
        try:
            fecha = datetime.fromisoformat(entry["created_at"]).strftime("%d/%m/%Y %H:%M")
        except ValueError:
            fecha = entry["created_at"]
        st.markdown(f"**{entry['usertag']}** · {fecha}")
        st.write("")
