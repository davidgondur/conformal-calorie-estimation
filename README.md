# Conformal Calorie Estimation (Nutrition5k)

Estimación de las **calorías de un plato** a partir de una foto, con **intervalos de predicción con garantía de cobertura** (Conformal Prediction). Un Vision Transformer exportado a **ONNX** hace la inferencia directamente en el navegador.

> Portfolio — [davidgondur.github.io](https://davidgondur.github.io)

## Qué muestra

- **Regresión conformal**: en lugar de "este plato tiene 520 kcal", la respuesta es "entre 410 y 680 kcal, con un 90 % de garantía".
- **Intervalos adaptativos**: comparación entre split conformal básico (ancho fijo), conformal normalizado y CQR (Conformalized Quantile Regression). Una ensalada sencilla recibe un intervalo estrecho y un guiso opaco uno ancho.
- **Dónde duda el modelo**: relación entre el ancho del intervalo y las características del plato (masa total, número de ingredientes).
- **Efecto de la cuantización**: ¿pasar el modelo ONNX de fp32 a int8 rompe la cobertura o solo ensancha los intervalos?
- **Demo interactiva sin servidor**: inferencia en el navegador con `onnxruntime-web` y un slider de α que recalcula el intervalo al instante.

## Stack

- **Modelo**: ViT pequeño (DeiT-tiny o DINOv2-small) con fine-tuning en PyTorch.
- **Exportación**: ONNX (fp32 e int8).
- **Conformal**: calibración offline en Python (MAPIE / TorchCP / implementación propia).
- **Web**: Quarto + Observable JS + `onnxruntime-web`, publicada en GitHub Pages.
- **Análisis** (opcional): notebooks de marimo para los experimentos de cobertura.

## Dataset

[Nutrition5k](https://github.com/google-research-datasets/Nutrition5k) (Google Research, CVPR 2021): unos 5.000 platos reales fotografiados en cafeterías de Google, con calorías, masa y macronutrientes medidos con precisión. Licencia CC BY 4.0.

Las imágenes no se suben al repositorio por su tamaño; hay que descargarlas desde la fuente oficial.

> Thames, Q. et al. *Nutrition5k: Towards Automatic Nutritional Understanding of Generic Food.* CVPR 2021.

## Estado

En desarrollo.

## Licencia

Código: por definir. Dataset: CC BY 4.0 (Google Research).
