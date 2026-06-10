# Low-Light Image Enhancement

Proyecto de investigación para la materia **Redes Neuronales** enfocado en mejorar la calidad visual de imágenes capturadas en condiciones de baja iluminación usando redes neuronales profundas.

**Dataset:** LOL-v1 + LOL-v2-real (1174 pares) — split 80/10/10 (seed 42): train=939, val=117, test=118.

---

## Modelos

### 1. RetinexNet + WGAN-GP

Fine-tuning adversarial de RetinexNet con un critic (discriminador) PatchGAN y Gradient Penalty (WGAN-GP).

La imagen oscura se descompone en **Reflectance (R)** e **Illumination (I)** via DecomNet. RelightNet realza la iluminación produciendo `I_enh`, y la salida final es `R × I_enh`. Un crítico PatchGAN (~2.76M params) guía al generador con loss WGAN-GP + reconstrucción L1.

- **Parámetros generador:** 555k | **Crítico:** 2.76M
- **Entrenamiento:** 50 épocas, batch=8, N_CRITIC=5, lr=1e-4
- **Resultados (val):** PSNR = 17.75 dB | SSIM = 0.669

![alt text](/diagramas/Retinex-GAN.png)

---

### 2. RetinexNet + Denoise-Net end-to-end

Extensión de RetinexNet que reemplaza el BM3D clásico del paper original con una **DenoiseNet entrenable end-to-end** (DnCNN residual liviana), permitiendo que el modelo aprenda la distribución de ruido real del dataset.

Tres sub-redes entrenadas conjuntamente: DecomNet (207k) + EnhanceNet/RelightNet con skip connections (237k) + DenoiseNet (225k). La loss combina reconstrucción, consistencia de reflectancia, suavidad de iluminación y pérdida perceptual VGG16.

- **Parámetros totales:** 670k
- **Entrenamiento:** 100 épocas, batch=8 (patches 128²), lr=1e-4, CosineAnnealingLR
- **Resultados (test):** PSNR = 17.14 dB | SSIM = 0.744

![alt text](/diagramas/RetinexNet-DenoiseNet.png)

---

### 3. Zero-DCE-FT++ *(mejor modelo)*

Fine-tuning de Zero-DCE (estimación de curvas de iluminación **sin ground truth**) con 4 mejoras sobre la versión base: pérdida perceptual VGG16, MiniDenoiser DnCNN entrenable, EMA de pesos (β=0.999) y Test-Time Augmentation D4 (8 transformaciones geométricas).

La imagen oscura entra a DCE-Net (79k params) que estima 8 mapas de curvas de iluminación (LE). La salida pasa por el MiniDenoiser (30k params) para eliminar artefactos.

- **Parámetros totales:** 109k
- **Entrenamiento:** lr adaptativo por grupos (DCE=5e-5, Denoiser=5e-4), eval sobre copia EMA
- **Resultados (test):** PSNR = 18.03 dB | SSIM = 0.733 | LPIPS = 0.283

![Arquitectura Zero-DCE-FT](/diagramas/ZeroDCE.png)

---

## Comparativa de resultados (test set)

| Modelo | PSNR (dB) | SSIM |
|---|---|---|
| RetinexNet base (preentrenado) | 15.72 | 0.508 |
| RetinexNet + WGAN-GP | 17.97 |0.7193|
| RetinexNet + Denoise-Net e2e | 17.14 | 0.744 |
| **Zero-DCE-FT++** | **18.03** | **0.733** |
