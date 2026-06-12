"""Acceso a la biblioteca compartida de imágenes (Supabase Storage + Postgres).

Cuando el usuario decide compartir, se suben la imagen original y la mejorada
al bucket `library` y se registra una fila en la tabla `library_images` con el
usertag. Todo se hace server-side con la service_role key leída de st.secrets.
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


def _to_png_bytes(image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def upload_to_library(original, enhanced, usertag: str) -> None:
    """Sube el par original/mejorada al bucket y registra la fila con el usertag.

    Lanza una excepción si la subida o el insert fallan (el caller decide
    cómo mostrarla al usuario).
    """
    client = get_client()
    storage = client.storage.from_(BUCKET)

    key = uuid.uuid4().hex
    enhanced_path = f"{key}.png"
    original_path = f"{key}_original.png"

    storage.upload(
        enhanced_path,
        _to_png_bytes(enhanced),
        file_options={"content-type": "image/png"},
    )
    storage.upload(
        original_path,
        _to_png_bytes(original),
        file_options={"content-type": "image/png"},
    )
    client.table(TABLE).insert(
        {
            "usertag": usertag.strip(),
            "image_path": enhanced_path,
            "original_path": original_path,
        }
    ).execute()


def fetch_library() -> list[dict]:
    """Devuelve las entradas de la biblioteca (más recientes primero).

    Cada entrada incluye `usertag`, `created_at`, `image_url` (mejorada) y
    `original_url` (None para entradas viejas sin original guardada).
    """
    client = get_client()
    rows = (
        client.table(TABLE)
        .select("usertag, image_path, original_path, created_at")
        .order("created_at", desc=True)
        .execute()
        .data
    )
    storage = client.storage.from_(BUCKET)
    for row in rows:
        row["image_url"] = storage.get_public_url(row["image_path"])
        row["original_url"] = (
            storage.get_public_url(row["original_path"])
            if row.get("original_path")
            else None
        )
    return rows
