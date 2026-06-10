"""Funciones auxiliares para la app de Low-Light Enhancement.

Contiene:
- La arquitectura Zero-DCE-FT++ (DCENet + MiniDenoiser), idéntica a la del
  notebook de entrenamiento `dev/models/02c_ZeroDCE_finetuning.ipynb`.
- Carga del modelo desde `dev/modelo.pth`.
- Preprocesamiento idéntico al usado en validación/test: RGB, float32 / 255,
  tensor (1, 3, H, W) en [0, 1]. Sin resize fijo ni normalización ImageNet.
- Inferencia con o sin Test-Time Augmentation (grupo diédrico D4).
- Postprocesamiento: tensor [0, 1] -> imagen PIL.
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

# Pesos del modelo final (EMA), versionados en el repo (~0.45 MB)
MODEL_PATH = Path(__file__).resolve().parent.parent / "dev" / "modelo.pth"

# Lado máximo de la imagen de entrada. Imágenes más grandes se redimensionan
# para mantener tiempos de inferencia razonables en la CPU de Streamlit Cloud.
# Con TTA el costo se multiplica x8, así que se usa un límite menor.
MAX_SIDE = 1024
MAX_SIDE_TTA = 600


# ---------------------------------------------------------------------------
# Arquitectura (idéntica al notebook 02c)
# ---------------------------------------------------------------------------

class DCENet(nn.Module):
    """DCE-Net del paper Zero-DCE: estima n_iter mapas de curvas de iluminación."""

    def __init__(self, n_iter=8):
        super().__init__()
        self.n_iter = n_iter
        f = 32
        self.conv1 = nn.Conv2d(3,     f, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(f,     f, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(f,     f, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(f,     f, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(f * 2, f, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(f * 2, f, kernel_size=3, padding=1)
        self.conv7 = nn.Conv2d(f * 2, 3 * n_iter, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.relu(self.conv1(x))
        x2 = self.relu(self.conv2(x1))
        x3 = self.relu(self.conv3(x2))
        x4 = self.relu(self.conv4(x3))
        x5 = self.relu(self.conv5(torch.cat([x4, x3], dim=1)))
        x6 = self.relu(self.conv6(torch.cat([x5, x2], dim=1)))
        A = torch.tanh(self.conv7(torch.cat([x6, x1], dim=1)))
        A_maps = torch.split(A, 3, dim=1)
        enhanced = x
        for A_i in A_maps:
            enhanced = enhanced + A_i * enhanced * (1 - enhanced)
        return enhanced, A_maps


class MiniDenoiser(nn.Module):
    """DnCNN liviana con aprendizaje residual: predice el ruido, no la imagen."""

    def __init__(self, channels=32, depth=5):
        super().__init__()
        layers = [
            nn.Conv2d(3, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        ]
        for _ in range(depth - 2):
            layers += [
                nn.Conv2d(channels, channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
            ]
        layers += [nn.Conv2d(channels, 3, kernel_size=3, padding=1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        noise = self.net(x)
        return (x - noise).clamp(0, 1)


class ZeroDCE_PP(nn.Module):
    """Zero-DCE-FT++ = DCE-Net + MiniDenoiser end-to-end."""

    def __init__(self, n_iter=8, denoiser_channels=32, denoiser_depth=5):
        super().__init__()
        self.dce = DCENet(n_iter=n_iter)
        self.denoiser = MiniDenoiser(channels=denoiser_channels, depth=denoiser_depth)

    def forward(self, x):
        enhanced, A_maps = self.dce(x)
        denoised = self.denoiser(enhanced.clamp(0, 1))
        return denoised, enhanced, A_maps


# ---------------------------------------------------------------------------
# Carga del modelo
# ---------------------------------------------------------------------------

def load_model(path=MODEL_PATH):
    """Carga Zero-DCE-FT++ (pesos EMA) en CPU y modo eval.

    Devuelve (modelo, metadata) donde metadata es el dict del checkpoint sin
    los pesos (métricas de test, notas de preprocesamiento, etc.).
    """
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = ZeroDCE_PP(n_iter=8, denoiser_channels=32, denoiser_depth=5)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    metadata = {k: v for k, v in checkpoint.items() if k != "state_dict"}
    return model, metadata


# ---------------------------------------------------------------------------
# Pre / post procesamiento
# ---------------------------------------------------------------------------

def preprocess(pil_img, max_side=MAX_SIDE):
    """PIL -> tensor (1, 3, H, W) en [0, 1]. Idéntico al pipeline de test.

    Convierte a RGB (descarta alfa / escala de grises) y, si la imagen supera
    max_side en su lado mayor, la reduce manteniendo la relación de aspecto.
    """
    img = pil_img.convert("RGB")
    if max(img.size) > max_side:
        scale = max_side / max(img.size)
        new_size = (round(img.width * scale), round(img.height * scale))
        img = img.resize(new_size, Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


def postprocess(tensor):
    """Tensor (1, 3, H, W) o (3, H, W) en [0, 1] -> imagen PIL."""
    if tensor.dim() == 4:
        tensor = tensor[0]
    arr = (tensor.permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr)


# ---------------------------------------------------------------------------
# Inferencia
# ---------------------------------------------------------------------------

@torch.no_grad()
def model_tta(model, low):
    """Test-Time Augmentation: promedia las predicciones sobre las 8 simetrías
    del cuadrado (grupo diédrico D4). Misma implementación que en test."""
    outputs = []
    for flip_h in [False, True]:
        for flip_v in [False, True]:
            for transpose in [False, True]:
                x = low
                if flip_h:
                    x = torch.flip(x, dims=[-1])
                if flip_v:
                    x = torch.flip(x, dims=[-2])
                if transpose:
                    # contiguous(): los tensores no contiguos ralentizan mucho
                    # las convoluciones en CPU
                    x = x.transpose(-1, -2).contiguous()

                out, _, _ = model(x)

                # Des-transformar (orden inverso)
                if transpose:
                    out = out.transpose(-1, -2)
                if flip_v:
                    out = torch.flip(out, dims=[-2])
                if flip_h:
                    out = torch.flip(out, dims=[-1])
                outputs.append(out)

    return torch.stack(outputs, dim=0).mean(dim=0)


@torch.no_grad()
def enhance(model, low, use_tta=True):
    """Ejecuta la inferencia. Con TTA (mejor calidad, ~8x más lenta) o sin."""
    if use_tta:
        return model_tta(model, low)
    out, _, _ = model(low)
    return out


def enhance_image(model, pil_img, use_tta=False):
    """Pipeline completo: PIL oscura -> PIL mejorada.

    Con TTA la imagen se procesa a resolución reducida (MAX_SIDE_TTA) para
    que el tiempo de inferencia en CPU siga siendo razonable.
    """
    max_side = MAX_SIDE_TTA if use_tta else MAX_SIDE
    x = preprocess(pil_img, max_side=max_side)
    pred = enhance(model, x, use_tta=use_tta)
    return postprocess(pred)
