"""
Contador reutilizable de parametros y FLOPs (por forward, batch=1) para modelos Keras y
PyTorch -- aplicado a varios de los modelos ya definidos en ../1-sgd-vs-adam/ y a la LSTM de
lstm_mantenimiento_predictivo.py, para comparar el coste computacional real de cada
arquitectura y no solo su accuracy -- relevante para decidir que modelo desplegar cuando los
recursos (CPU embebida, latencia de inferencia) son una restriccion real, no solo un detalle
academico.

Convencion de conteo: una multiplicacion-acumulacion = 2 FLOPs (estandar en la literatura),
para las capas con pesos presentes en este repo (Dense/Linear, Conv2D/Conv2d, LSTM); el resto
de capas (Dropout, Flatten, MaxPooling, activaciones) se cuenta como 0 FLOPs -- su coste es
marginal frente a las capas con pesos y no es el objetivo de esta comparacion.

Uso: python model_complexity.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from _common import cargar_modulo

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


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
    cada submodulo con pesos durante una unica pasada con entrada_ejemplo (batch=1)."""
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

    for modulo in model.modules():
        if isinstance(modulo, nn.Linear):
            hooks.append(modulo.register_forward_hook(hook_linear))
        elif isinstance(modulo, nn.Conv2d):
            hooks.append(modulo.register_forward_hook(hook_conv2d))
        elif isinstance(modulo, nn.LSTM):
            hooks.append(modulo.register_forward_hook(hook_lstm))

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

    dense_tf = cargar_modulo("sgd_vs_adam_dense_tf.py")
    resultados.append(analizar_modelo_keras(
        dense_tf.build_model(tf.keras.optimizers.Adam()), "dense_mnist (Keras)"))

    cnn_tf = cargar_modulo("sgd_vs_adam_cnn_tf.py")
    resultados.append(analizar_modelo_keras(
        cnn_tf.build_model(tf.keras.optimizers.Adam()), "cnn_fashion_mnist (Keras)"))

    customer_tf = cargar_modulo("sgd_vs_adam_customer_tf.py")
    resultados.append(analizar_modelo_keras(
        customer_tf.build_model(tf.keras.optimizers.Adam()), "dense_tipos_clientes (Keras)"))

    spiral_tf = cargar_modulo("sgd_vs_adam_spiral_tf.py")
    resultados.append(analizar_modelo_keras(
        spiral_tf.build_model(tf.keras.optimizers.Adam()), "dense_zonas_espirales (Keras)"))

    import importlib.util
    ruta_lstm = Path(__file__).parent / "lstm_mantenimiento_predictivo.py"
    spec = importlib.util.spec_from_file_location("lstm_mantenimiento_predictivo", ruta_lstm)
    lstm_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lstm_mod)
    resultados.append(analizar_modelo_keras(lstm_mod.build_model(), "lstm_mantenimiento_predictivo (Keras)"))

    dense_torch = cargar_modulo("sgd_vs_adam_dense_torch.py")
    m = dense_torch.build_model()
    resultados.append(analizar_modelo_torch(
        m, torch.zeros(1, 28 * 28, device=dense_torch.DEVICE), "dense_mnist (PyTorch)"))

    cnn_torch = cargar_modulo("sgd_vs_adam_cnn_torch.py")
    m = cnn_torch.build_model()
    resultados.append(analizar_modelo_torch(
        m, torch.zeros(1, 1, 28, 28, device=cnn_torch.DEVICE), "cnn_fashion_mnist (PyTorch)"))

    customer_torch = cargar_modulo("sgd_vs_adam_customer_torch.py")
    m = customer_torch.build_model()
    resultados.append(analizar_modelo_torch(
        m, torch.zeros(1, 2, device=customer_torch.DEVICE), "dense_tipos_clientes (PyTorch)"))

    spiral_torch = cargar_modulo("sgd_vs_adam_spiral_torch.py")
    m = spiral_torch.build_model()
    resultados.append(analizar_modelo_torch(
        m, torch.zeros(1, 2, device=spiral_torch.DEVICE), "dense_zonas_espirales (PyTorch)"))

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
