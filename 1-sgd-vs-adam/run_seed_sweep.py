"""
Barrido multi-semilla (esquema "Actual": seed_split y seed_modelo sorteados de forma
INDEPENDIENTE en cada repeticion, no atados entre si, no en rejilla -- estimacion Monte Carlo
estandar de la media y la varianza del resultado frente a la semilla) sobre las 8 unidades
sgd_vs_adam de este proyecto (denso/CNN x full-batch/mini-batch x TF/PyTorch). Mismo patron
que numpy-neural-networks-from-scratch/run_seed_sweep.py -- ver su docstring y el README raiz
de ese repo ("Metodologia: por que tres semillas") para la discusion completa de por que este
muestreo y no una rejilla.

Escribe results_sgd_vs_adam/<unidad>/metrics_seed_sweep.json y seed_sweep_curvas.png, sin
tocar el metrics.json de la ejecucion canonica ya documentada (seed_split=42, seed_modelo=42).

Uso:
    python run_seed_sweep.py                         # las 8 unidades, N por defecto
    python run_seed_sweep.py --framework tf --n 20    # solo las 4 unidades TF
    python run_seed_sweep.py --solo cnn_tf_fullbatch --n 3
"""

import argparse
import importlib.util
import json
import statistics
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).parent
N_SEEDS_DEFAULT = 20
COLORES_VARIANTES = {"sgd": "#4C72B0", "adam": "#55A868"}

# (nombre_unidad, script, framework)
UNIDADES = [
    ("dense_tf_fullbatch", "sgd_vs_adam_dense_tf.py", "tf"),
    ("dense_tf_minibatch", "sgd_vs_adam_dense_tf_full.py", "tf"),
    ("cnn_tf_fullbatch", "sgd_vs_adam_cnn_tf.py", "tf"),
    ("cnn_tf_minibatch", "sgd_vs_adam_cnn_tf_full.py", "tf"),
    ("dense_torch_fullbatch", "sgd_vs_adam_dense_torch.py", "torch"),
    ("dense_torch_minibatch", "sgd_vs_adam_dense_torch_full.py", "torch"),
    ("cnn_torch_fullbatch", "sgd_vs_adam_cnn_torch.py", "torch"),
    ("cnn_torch_minibatch", "sgd_vs_adam_cnn_torch_full.py", "torch"),
    ("customer_tf", "sgd_vs_adam_customer_tf.py", "tf"),
    ("customer_torch", "sgd_vs_adam_customer_torch.py", "torch"),
    ("spiral_tf", "sgd_vs_adam_spiral_tf.py", "tf"),
    ("spiral_torch", "sgd_vs_adam_spiral_torch.py", "torch"),
]


def cargar_modulo(nombre, ruta):
    carpeta = str(ruta.parent)
    if carpeta not in sys.path:
        sys.path.insert(0, carpeta)
    spec = importlib.util.spec_from_file_location(nombre, ruta)
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


def semillas_independientes(n, offset):
    rng = np.random.default_rng(1_000_000 + offset)
    splits = rng.integers(0, 2**31 - 1, size=n)
    modelos = rng.integers(0, 2**31 - 1, size=n)
    return list(zip(splits.tolist(), modelos.tolist()))


def metrica_variantes(metrics):
    return {nombre: datos["accuracy_test"] for nombre, datos in metrics["resultados"].items()}


def historial_variantes(metrics):
    return {nombre: datos["historial_loss_val"] for nombre, datos in metrics["resultados"].items()}


def resumir(valores):
    return {
        "n": len(valores),
        "media": statistics.fmean(valores),
        "desviacion_tipica": statistics.stdev(valores) if len(valores) > 1 else 0.0,
        "min": min(valores),
        "max": max(valores),
        "valores": valores,
    }


def graficar_curvas_superpuestas(por_variante_historial, titulo, ruta_salida):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for variante, historiales in por_variante_historial.items():
        color = COLORES_VARIANTES.get(variante, "#4C72B0")
        for i, historial in enumerate(historiales):
            ax.plot(range(1, len(historial) + 1), historial, color=color, alpha=0.55, linewidth=1.1,
                    label=variante.upper() if i == 0 else None)
    ax.set_yscale("log")
    ax.set_xlabel("Epoca")
    ax.set_ylabel("Perdida de validacion (escala log)")
    ax.set_title(titulo, fontweight="bold", fontsize=11)
    ax.grid(True, which="both", linestyle="--", alpha=0.35)
    ax.legend(fontsize=9)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    plt.tight_layout()
    Path(ruta_salida).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(ruta_salida, dpi=150)
    plt.close()


def ejecutar_unidad(nombre, script, n, offset, sufijo="", seeds_fn=None):
    """sufijo="" -> esquema Actual (semillas_independientes). sufijo="_esquemaB" -> lo
    invoca run_seed_sweep_esquemaB.py, que reutiliza esta funcion con su propio seeds_fn."""
    ruta_absoluta = ROOT / script
    modulo = cargar_modulo(f"sweep_{nombre}{sufijo}", ruta_absoluta)

    pares = (seeds_fn or semillas_independientes)(n, offset)
    por_variante = {}
    por_variante_historial = {}
    t0 = time.time()
    for i, (seed_split, seed_modelo) in enumerate(pares):
        metrics = modulo.main(seed_split=int(seed_split), seed_modelo=int(seed_modelo),
                               quiet=True, guardar_graficas=False)
        resultado = metrica_variantes(metrics)
        for variante, valor in resultado.items():
            por_variante.setdefault(variante, []).append(valor)
        for variante, historial in historial_variantes(metrics).items():
            por_variante_historial.setdefault(variante, []).append(historial)
        print(f"  [{nombre}{sufijo}] {i + 1}/{n} (seed_split={seed_split}, seed_modelo={seed_modelo}) -- "
              f"{time.time() - t0:.0f}s acumulados", flush=True)

    resumen = {variante: resumir(valores) for variante, valores in por_variante.items()}
    carpeta_resultados = ROOT / "results_sgd_vs_adam" / nombre
    carpeta_resultados.mkdir(parents=True, exist_ok=True)
    salida = carpeta_resultados / f"metrics_seed_sweep{sufijo}.json"
    salida.write_text(json.dumps(resumen, indent=2, ensure_ascii=False), encoding="utf-8")

    ruta_curvas = carpeta_resultados / f"seed_sweep_curvas{sufijo}.png"
    etiqueta_esquema = " (esquema B, split fijo)" if sufijo else ""
    graficar_curvas_superpuestas(por_variante_historial, f"{nombre}{etiqueta_esquema}: "
                                  "perdida de validacion por semilla", ruta_curvas)

    print(f"[{nombre}{sufijo}] guardado en {salida} y {ruta_curvas} ({time.time() - t0:.0f}s total)", flush=True)
    return resumen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=N_SEEDS_DEFAULT)
    parser.add_argument("--solo", type=str, default=None, help="nombre de una sola unidad a ejecutar")
    parser.add_argument("--framework", type=str, default=None, choices=["tf", "torch"],
                         help="filtra solo las unidades de ese framework")
    args = parser.parse_args()

    unidades = UNIDADES
    if args.solo is not None:
        unidades = [u for u in unidades if u[0] == args.solo]
    if args.framework is not None:
        unidades = [u for u in unidades if u[2] == args.framework]
    if not unidades:
        print(f"Ninguna unidad coincide con --solo={args.solo} --framework={args.framework}. "
              f"Disponibles: {[u[0] for u in UNIDADES]}")
        sys.exit(1)

    for offset, (nombre, script, framework) in enumerate(unidades):
        print(f"\n=== Unidad: {nombre} (N={args.n}) ===", flush=True)
        ejecutar_unidad(nombre, script, args.n, offset)


if __name__ == "__main__":
    main()
