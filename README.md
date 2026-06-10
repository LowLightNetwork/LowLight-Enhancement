# Low-Light Image Enhancement
Proyecto de investigación para la materia Redes Neuronales enfocado en mejorar la calidad visual de imágenes capturadas en condiciones de baja iluminación usando redes neuronales profundas.

Dataset: LOL-v1 + LOL-v2-real (1174 pares) — split 80/10/10 (seed 42): train=939, val=117, test=118.

---

## Modelos

### 1. RetinexNet + WGAN-GP

Fine-tuning adversarial de RetinexNet con un critic (discriminador) PatchGAN y Gradient Penalty (WGAN-GP).

La imagen oscura se descompone en Reflectance (R) e Illumination (I) via DecomNet. RelightNet realza la iluminación produciendo I_delta, y la salida final es R × I_delta. Un crítico PatchGAN guía al generador con loss WGAN-GP + reconstrucción L1. Los pesos base se cargan desde el repositorio `aasharma90/RetinexNet_PyTorch`.

- Parámetros generador: 555k | Crítico: 2.76M
- Entrenamiento: 50 épocas, batch=8, N_CRITIC=5, GP_LAMBDA=10, L1_WEIGHT=1.0, ADV_WEIGHT=0.01, lr=1e-4 (Adam, betas=(0.5, 0.9))
- Resultados (test): PSNR = 17.12 dB | SSIM = 0.6323

![Arquitectura RetinexNet GAN](/diagramas/Retinex-GAN.png)

---

### 2. RetinexNet + Denoise-Net end-to-end

Extensión de RetinexNet que reemplaza el BM3D clásico del paper original con una DenoiseNet entrenable end-to-end (DnCNN residual liviana), permitiendo que el modelo aprenda la distribución de ruido real del dataset.

Tres sub-redes entrenadas conjuntamente: DecomNet (207k) + EnhanceNet con skip connections (237k) + DenoiseNet (225k). La loss combina reconstrucción L1, consistencia de reflectancia, suavidad de iluminación structure-aware, pérdida perceptual VGG16, (1−SSIM) y consistencia de color (coseno RGB por píxel).

- Parámetros totales: 670k
- Entrenamiento: 50 épocas, batch=8 (patches 128²), lr=2e-4 (Adam), warmup lineal 5 épocas → CosineAnnealingLR
- Resultados (test): PSNR = 16.93 dB | SSIM = 0.8124

![Arquitectura RetinexNet DenoiseNet](/diagramas/RetinexNet-DenoiseNet.png)

---

### 3. Zero-DCE-FT++ (mejor PSNR)

Fine-tuning de Zero-DCE (estimación de curvas de iluminación sin ground truth) con 4 mejoras sobre la versión base: pérdida perceptual VGG16, MiniDenoiser DnCNN entrenable, EMA de pesos (β=0.999) y Test-Time Augmentation D4 (8 transformaciones geométricas).

La imagen oscura entra a DCE-Net (79k params) que estima 8 mapas de curvas de iluminación (LE). La salida pasa por el MiniDenoiser (30k params) para eliminar artefactos. Los pesos de DCE-Net se inicializan desde el checkpoint oficial (`Epoch99.pth`); el MiniDenoiser se entrena desde cero. El modelo se evalúa sobre la copia EMA durante el entrenamiento y con TTA en test.

- Parámetros totales: 109k
- Entrenamiento: lr adaptativo por grupos (DCE=5e-5, Denoiser=5e-4), weight_decay=1e-4, grad_clip=1.0, CosineAnnealingLR, EMA β=0.999
- Loss: L1 + 0.5·(1−SSIM) + 0.1·VGG_perceptual + 0.1·L_zdce (regularización no-referencial)
- Resultados (test, con TTA): PSNR = 18.03 dB | SSIM = 0.733 | LPIPS = 0.283

![Arquitectura Zero-DCE-FT](/diagramas/ZeroDCE.png)

---

## Comparativa de resultados

> Todos los modelos están evaluados sobre el conjunto de **test**.

| Modelo | Conjunto | PSNR (dB) | SSIM |
|---|---|---|---|
| RetinexNet base (preentrenado) | test | 15.81 | 0.5278 |
| RetinexNet + WGAN-GP | test | 17.12 | 0.6323 |
| RetinexNet + Denoise-Net e2e | test | 16.93 | 0.8124 |
| Zero-DCE-FT++ (con TTA) | test | 18.03 | 0.733 |

**Notas sobre los resultados:**
- Zero-DCE-FT++ obtiene el mejor PSNR (18.03 dB).
- RetinexNet + Denoise-Net obtiene el mejor SSIM (0.8124), lo que indica mayor similitud estructural percibida.
- RetinexNet + WGAN-GP mejora +1.31 dB y +0.1045 SSIM respecto al modelo base sobre el mismo conjunto de test.
