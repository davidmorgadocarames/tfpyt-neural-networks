"""
Contador reutilizable de parametros y FLOPs (por forward, batch=1) para los 10 `baseline_*.py`
de este apartado (5 problemas x 2 frameworks) -- version de este apartado de
../2-metodologia-ml/model_complexity.py, extendida para contar tambien el coste de las capas
BatchNorm (07 y 08), ausentes del contador anterior porque ninguno de los modelos de
../1-sgd-vs-adam/ las usaba.

Convencion de conteo: una multiplicacion-acumulacion = 2 FLOPs (estandar en la literatura) para
las capas con pesos (Dense/Linear, Conv2D/Conv2d, LSTM); BatchNorm cuenta 4 FLOPs por activacion
normalizada (restar la media, dividir por la desviacion tipica, escalar por gamma, desplazar por
beta -- 4 operaciones elementales, tanto en `tf.keras.layers.BatchNormalization` como en
`torch.nn.BatchNorm1d`/`BatchNorm2d`); el resto de capas (Dropout, Flatten, MaxPooling,
activaciones puras) se cuenta como 0 FLOPs -- su coste es marginal frente a las capas con pesos
y no es el objetivo de esta comparacion.

Uso: python model_complexity.py
"""

import importlib.util
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

PROBLEMAS = [
    "03-tipos-clientes",
    "04-prediccion-temperatura-dia-noche",
    "06-zonas-espirales",
    "07-reconocimiento-digitos",
    "08-cnn-fashion-mnist",
]


COMPARTIDOS = ["adam", "lr_decay", "weight_decay", "baseline_tf", "baseline_torch"]


def cargar_modulo(problema, nombre_script):
    """Los 5 problemas comparten nombres de fichero (adam.py, baseline_tf.py, etc.) -- antes de
    cargar cada uno, se retira del sys.path la carpeta de cualquier OTRO problema y se purgan del
    cache de sys.modules los nombres compartidos, para que los imports internos de cada script
    (`from adam import ...`, `from baseline_tf import ...`) resuelvan siempre dentro de su propia
    carpeta y no reutilicen por error el modulo de otro problema ya cargado antes."""
    carpeta = str(Path(__file__).parent / problema)
    for otro in PROBLEMAS:
        otra_carpeta = str(Path(__file__).parent / otro)
        if otra_carpeta in sys.path:
            sys.path.remove(otra_carpeta)
    sys.path.insert(0, carpeta)
    for nombre in COMPARTIDOS:
        sys.modules.pop(nombre, None)

    ruta = Path(carpeta) / nombre_script
    nombre_modulo = f"{problema}_{nombre_script}".replace(".py", "").replace("-", "_")
    spec = importlib.util.spec_from_file_location(nombre_modulo, ruta)
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


# ---------------------------------------------------------------- Keras ----
def _flops_capa_keras(layer) -> int:
    import tensorflow as tf

    if isinstance(layer, tf.keras.layers.Dense):
        in_dim = layer.kernel.shape[0]
        out_dim = layer.units
        flops = 2 * in_dim * out_dim
        if layer.use_bias:
            flops += out_dim
        return int(flops)

    if isinstance(layer, tf.keras.layers.Conv2D):
        kh, kw, in_c, out_c = layer.kernel.shape
        _, out_h, out_w, _ = layer.output.shape
        flops = 2 * kh * kw * in_c * out_c * out_h * out_w
        if layer.use_bias:
            flops += out_h * out_w * out_c
        return int(flops)

    if isinstance(layer, tf.keras.layers.LSTM):
        in_dim = layer.input.shape[-1]
        timesteps = layer.input.shape[1]
        hidden = layer.units
        # 4 puertas (input/forget/cell/output), cada una una matriz (in_dim+hidden) x hidden,
        # mas bias -- FLOPs por paso temporal, multiplicado por el numero de pasos de entrada.
        flops_por_paso = 2 * 4 * hidden * (in_dim + hidden + 1)
        return int(flops_por_paso * timesteps)

    if isinstance(layer, tf.keras.layers.BatchNormalization):
        activaciones = 1
        for dim in layer.output.shape[1:]:
            activaciones *= int(dim)
        return int(4 * activaciones)

    return 0


def analizar_modelo_keras(model, nombre) -> dict:
    total_params = model.count_params()
    total_flops = sum(_flops_capa_keras(layer) for layer in model.layers)
    return {"nombre": nombre, "framework": "keras", "parametros": int(total_params),
            "flops_forward": int(total_flops)}


# -------------------------------------------------------------- PyTorch ----
def analizar_modelo_torch(model, entrada_ejemplo, nombre) -> dict:
    """A diferencia de Keras, los modulos de PyTorch no exponen la forma de su salida sin
    ejecutar un forward real -- se registran forward hooks que capturan la forma de salida de
    cada submodulo con pesos (o normalizador) durante una unica pasada con entrada_ejemplo
    (batch=1)."""
    import torch
    import torch.nn as nn

    total_params = sum(p.numel() for p in model.parameters())
    flops_acumulados = {"total": 0}
    hooks = []

    def hook_linear(module, inputs, output):
        flops = 2 * module.in_features * module.out_features
        if module.bias is not None:
            flops += module.out_features
        flops_acumulados["total"] += flops

    def hook_conv2d(module, inputs, output):
        _, out_c, out_h, out_w = output.shape
        kh, kw = module.kernel_size
        in_c = module.in_channels // module.groups
        flops = 2 * kh * kw * in_c * out_c * out_h * out_w
        if module.bias is not None:
            flops += out_h * out_w * out_c
        flops_acumulados["total"] += flops

    def hook_lstm(module, inputs, output):
        x = inputs[0]
        timesteps = x.shape[1] if module.batch_first else x.shape[0]
        flops_por_paso = 2 * 4 * module.hidden_size * (module.input_size + module.hidden_size + 1)
        flops_acumulados["total"] += flops_por_paso * timesteps * module.num_layers

    def hook_batchnorm(module, inputs, output):
        activaciones = 1
        for dim in output.shape[1:]:
            activaciones *= int(dim)
        flops_acumulados["total"] += 4 * activaciones

    for modulo in model.modules():
        if isinstance(modulo, nn.Linear):
            hooks.append(modulo.register_forward_hook(hook_linear))
        elif isinstance(modulo, nn.Conv2d):
            hooks.append(modulo.register_forward_hook(hook_conv2d))
        elif isinstance(modulo, nn.LSTM):
            hooks.append(modulo.register_forward_hook(hook_lstm))
        elif isinstance(modulo, (nn.BatchNorm1d, nn.BatchNorm2d)):
            hooks.append(modulo.register_forward_hook(hook_batchnorm))

    model.eval()
    with torch.no_grad():
        model(entrada_ejemplo)
    for h in hooks:
        h.remove()

    return {"nombre": nombre, "framework": "pytorch", "parametros": int(total_params),
            "flops_forward": int(flops_acumulados["total"])}


def main(quiet=False) -> dict:
    import tensorflow as tf
    import torch

    resultados = []

    # -------- 03: tipos de clientes (2 -> 5 -> 3) --------
    m03_tf = cargar_modulo("03-tipos-clientes", "baseline_tf.py")
    resultados.append(analizar_modelo_keras(m03_tf.build_model(), "03-tipos-clientes (Keras)"))
    m03_torch = cargar_modulo("03-tipos-clientes", "baseline_torch.py")
    modelo = m03_torch.build_model()
    resultados.append(analizar_modelo_torch(
        modelo, torch.zeros(1, 2, device=m03_torch.DEVICE), "03-tipos-clientes (PyTorch)"))

    # -------- 04: temperatura dia/noche (LSTM(32) -> 16 -> 1) --------
    m04_tf = cargar_modulo("04-prediccion-temperatura-dia-noche", "baseline_tf.py")
    resultados.append(analizar_modelo_keras(m04_tf.build_model(), "04-temperatura-dia-noche (Keras)"))
    m04_torch = cargar_modulo("04-prediccion-temperatura-dia-noche", "baseline_torch.py")
    modelo = m04_torch.build_model()
    resultados.append(analizar_modelo_torch(
        modelo, torch.zeros(1, m04_tf.VENTANA, 1, device=m04_torch.DEVICE), "04-temperatura-dia-noche (PyTorch)"))

    # -------- 06: zonas de espirales (2 -> 64 -> 64 -> 3) --------
    m06_tf = cargar_modulo("06-zonas-espirales", "baseline_tf.py")
    resultados.append(analizar_modelo_keras(m06_tf.build_model(), "06-zonas-espirales (Keras)"))
    m06_torch = cargar_modulo("06-zonas-espirales", "baseline_torch.py")
    modelo = m06_torch.build_model()
    resultados.append(analizar_modelo_torch(
        modelo, torch.zeros(1, 2, device=m06_torch.DEVICE), "06-zonas-espirales (PyTorch)"))

    # -------- 07: digitos MNIST (Dense(128) -> BatchNorm -> LeakyReLU -> 10) --------
    m07_tf = cargar_modulo("07-reconocimiento-digitos", "baseline_tf.py")
    resultados.append(analizar_modelo_keras(m07_tf.build_model(), "07-reconocimiento-digitos (Keras)"))
    m07_torch = cargar_modulo("07-reconocimiento-digitos", "baseline_torch.py")
    modelo = m07_torch.build_model()
    resultados.append(analizar_modelo_torch(
        modelo, torch.zeros(1, 28 * 28, device=m07_torch.DEVICE), "07-reconocimiento-digitos (PyTorch)"))

    # -------- 08: CNN Fashion-MNIST --------
    m08_tf = cargar_modulo("08-cnn-fashion-mnist", "baseline_tf.py")
    resultados.append(analizar_modelo_keras(m08_tf.build_model(), "08-cnn-fashion-mnist (Keras)"))
    m08_torch = cargar_modulo("08-cnn-fashion-mnist", "baseline_torch.py")
    modelo = m08_torch.build_model()
    resultados.append(analizar_modelo_torch(
        modelo, torch.zeros(1, 1, 28, 28, device=m08_torch.DEVICE), "08-cnn-fashion-mnist (PyTorch)"))

    (RESULTS_DIR / "model_complexity_metrics.json").write_text(
        json.dumps({"resultados": resultados}, indent=2, ensure_ascii=False), encoding="utf-8")

    nombres = [r["nombre"] for r in resultados]
    parametros = [r["parametros"] for r in resultados]
    flops = [r["flops_forward"] for r in resultados]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    colores = ["#4C72B0" if "Keras" in n else "#55A868" for n in nombres]
    axes[0].barh(nombres, parametros, color=colores)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("Parametros (escala log)")
    axes[0].set_title("Parametros totales por modelo", fontweight="bold")
    axes[1].barh(nombres, flops, color=colores)
    axes[1].set_xscale("log")
    axes[1].set_xlabel("FLOPs por forward, batch=1 (escala log)")
    axes[1].set_title("Coste computacional por modelo", fontweight="bold")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "model_complexity.png", dpi=150)
    plt.close(fig)

    if not quiet:
        for r in resultados:
            print(f"{r['nombre']:38s} parametros={r['parametros']:>10,d}  flops={r['flops_forward']:>14,d}")
        print(f"Resultados guardados en {RESULTS_DIR}")

    return {"resultados": resultados}


if __name__ == "__main__":
    main()
