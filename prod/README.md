# App — Low-Light Image Enhancement (Zero-DCE-FT++)

Aplicación web (Streamlit) que toma una foto oscura y devuelve una versión más clara
usando el modelo **Zero-DCE-FT++** entrenado en este proyecto (mejor PSNR sobre test: 18.03 dB con TTA).

**App desplegada:** _(pendiente — agregar URL de Streamlit Cloud)_

## Estructura

| Archivo | Responsabilidad |
|---|---|
| `app.py` | Interfaz Streamlit: upload, controles, visualización lado a lado, descarga y compartir en la biblioteca |
| `pages/1_Biblioteca.py` | Galería pública con las imágenes compartidas por los usuarios |
| `library_db.py` | Acceso a Supabase (Storage + Postgres) para subir y listar imágenes de la biblioteca |
| `utils.py` | Arquitectura del modelo, carga de pesos, preprocesamiento, TTA y postprocesamiento |
| `requirements.txt` | Dependencias con versiones fijadas |

Los pesos del modelo (`dev/modelo.pth`, ~0.45 MB) están versionados en el repositorio,
por lo que no se necesita ninguna descarga externa.

## Correr localmente

Desde la **raíz del repositorio**:

```bash
pip install -r prod/requirements.txt
streamlit run prod/app.py
```

La app queda disponible en `http://localhost:8501`.

## Biblioteca de imágenes compartidas

La app incluye una página **Biblioteca** (navegación lateral) con las imágenes
mejoradas que los usuarios deciden compartir. Nada se sube por defecto: después
de mejorar una imagen, el usuario puede ingresar un nombre/usertag y apretar
**Subir a la biblioteca**. Solo se comparte la imagen mejorada (no la original).

La persistencia usa Supabase:

- **Storage**: bucket `library` (lectura pública) con los PNG.
- **Postgres**: tabla `library_images` (`usertag`, `image_path`, `created_at`),
  con RLS habilitado y lectura pública. El esquema ya está aplicado como
  migración (`create_library`) en el proyecto de Supabase.

### Credenciales

La app lee las credenciales desde `st.secrets`. Localmente, crear
`prod/.streamlit/secrets.toml` (está en `.gitignore`):

```toml
SUPABASE_URL = "https://ryejeoiccbmtxaafkjaz.supabase.co"
SUPABASE_KEY = "<service_role key>"
```

La `service_role` key se copia desde el dashboard de Supabase
(Settings → API). Solo se usa server-side; nunca llega al navegador.
En Streamlit Cloud, cargar los mismos dos valores en
App settings → Secrets.

Si faltan las credenciales, la app sigue funcionando normalmente y solo se
deshabilita la sección de compartir/galería.

## Detalles de inferencia

- **Preprocesamiento** (idéntico al de validación/test del entrenamiento):
  convertir a RGB → `float32 / 255` → tensor `(1, 3, H, W)` en `[0, 1]`.
  Sin normalización ImageNet. Imágenes con lado mayor a 1024 px se reducen
  manteniendo la relación de aspecto (para tiempos razonables en CPU).
- **TTA (Test-Time Augmentation):** promedia las predicciones sobre las 8 simetrías
  del cuadrado (grupo D4). Es la configuración que reporta el mejor PSNR, pero es
  ~8× más lenta en CPU. Por eso viene desactivada por defecto y, cuando se activa,
  la imagen se procesa a resolución reducida (máx. 600 px de lado).
- **Carga del modelo:** cacheada con `@st.cache_resource` para no recargar los pesos
  en cada interacción.

## Despliegue (Streamlit Cloud)

1. Repo público en GitHub con `prod/` y `dev/modelo.pth`.
2. En [share.streamlit.io](https://share.streamlit.io): New app → seleccionar el repo,
   branch `main`, main file `prod/app.py`.
3. En Advanced settings, elegir Python 3.11.
4. Streamlit Cloud detecta `prod/requirements.txt` automáticamente (está junto al main file).
