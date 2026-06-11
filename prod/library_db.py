"""Acceso a la biblioteca compartida de imágenes (Supabase Storage + Postgres).

Las imágenes mejoradas que el usuario decide compartir se suben al bucket
`library` y se registra una fila en la tabla `library_images` con el usertag.
Todo se hace server-side con la service_role key leída de st.secrets.
"""

import io
import uuid

import streamlit as st
from supabase import create_client

BUCKET = "library"
TABLE = "library_images"


def supabase_configured() -> bool:
    """Indica si los secrets de Supabase están disponibles y completos."""
    try:
        url = st.secrets["SUPABASE_URL"]
        key = st.secrets["SUPABASE_KEY"]
    except (KeyError, FileNotFoundError):
        return False
    return bool(url) and bool(key) and "PEGAR" not in key


@st.cache_resource
def get_client():
    """Cliente de Supabase, creado una sola vez por proceso."""
    return create_client(st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_KEY"])


def upload_to_library(image, usertag: str) -> None:
    """Sube una imagen PIL al bucket y registra la fila con el usertag.

    Lanza una excepción si la subida o el insert fallan (el caller decide
    cómo mostrarla al usuario).
    """
    client = get_client()

    buf = io.BytesIO()
    image.save(buf, format="PNG")
    path = f"{uuid.uuid4().hex}.png"

    client.storage.from_(BUCKET).upload(
        path, buf.getvalue(), file_options={"content-type": "image/png"}
    )
    client.table(TABLE).insert(
        {"usertag": usertag.strip(), "image_path": path}
    ).execute()


def fetch_library() -> list[dict]:
    """Devuelve las entradas de la biblioteca (más recientes primero).

    Cada entrada incluye `usertag`, `created_at` y `image_url` (URL pública).
    """
    client = get_client()
    rows = (
        client.table(TABLE)
        .select("usertag, image_path, created_at")
        .order("created_at", desc=True)
        .execute()
        .data
    )
    storage = client.storage.from_(BUCKET)
    for row in rows:
        row["image_url"] = storage.get_public_url(row["image_path"])
    return rows
