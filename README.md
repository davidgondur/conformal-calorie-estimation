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

Solo se usan las fotos cenitales RGB (unos 3.500 platos, ~1,4 GB), no los vídeos laterales que ocupan casi todo el archivo completo de 181 GB:

```bash
uv run n5k-download              # → data/nutrition5k/ (añade --depth para depth_raw.png)
```

### Particiones

En local solo se guardan las particiones **oficiales**, tal como las publica Nutrition5k (`data/nutrition5k/dish_ids/splits/depth_{train,test}_ids.txt`): **train (2.755 platos)** y **test (507)**. No se guarda en disco ninguna otra partición.

Validación, calibración o cualquier otra división se saca de `train` **dentro de cada script o notebook**, con una semilla explícita. Así las proporciones quedan junto al experimento que las usa:

```python
from conformal_calorie_estimation.data import (
    Nutrition5kOverhead, official_splits, read_dishes, split_by_groups,
)

root = "data/nutrition5k"
dishes = read_dishes(root)
official = official_splits(root)                  # {"train", "test"}
parts = split_by_groups(official["train"], dishes, {"val": 0.1, "calib": 0.2}, seed=0)
# parts = {"val": ~276, "calib": ~552, "train": ~1927}

train_ds = Nutrition5kOverhead(root, parts["train"], targets=("calories",), dishes=dishes)
```

Reglas:

- **`test` nunca se subdivide** y solo se usa al final, para medir la cobertura.
- **`calib` no puede influir en el modelo**, ni siquiera en el early stopping o en la elección de hiperparámetros: para eso está `val`. Si no, la garantía de cobertura conformal deja de valer.
- `split_by_groups` reparte **grupos de montaje incremental** enteros, para que los pasos casi idénticos de un mismo plato no queden en dos particiones a la vez. Los grupos se detectan con una heurística: dos platos consecutivos, a menos de 5 minutos, en los que los ingredientes del primero son un subconjunto de los del segundo. La agrupación oficial no está publicada.

> Thames, Q. et al. *Nutrition5k: Towards Automatic Nutritional Understanding of Generic Food.* CVPR 2021.

## Estado

En desarrollo.

## Licencia

Código: por definir. Dataset: CC BY 4.0 (Google Research).
