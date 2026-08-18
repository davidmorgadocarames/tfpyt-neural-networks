"""
Genera, para cada una de las 8 unidades sgd_vs_adam de este proyecto, un dot plot SGD vs Adam
(esquema Actual) y un dot plot de comparacion split libre vs split fijo (esquema B), a partir
de results_sgd_vs_adam/<unidad>/metrics_seed_sweep*.json (ver run_seed_sweep.py). Mismo
criterio que numpy-neural-networks-from-scratch/plot_seed_sweep.py: con semillas
independientes no hay ningun orden con sentido entre "semilla 3" y "semilla 4", asi que cada
valor se dibuja como un punto individual en vez de agregarlo en una barra o histograma.

Uso: python plot_seed_sweep.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).parent
UNIDADES = [
    "dense_tf_fullbatch", "dense_tf_minibatch", "cnn_tf_fullbatch", "cnn_tf_minibatch",
    "dense_torch_fullbatch", "dense_torch_minibatch", "cnn_torch_fullbatch", "cnn_torch_minibatch",
]
COLORES_VARIANTES = {"sgd": "#4C72B0", "adam": "#55A868"}


def _formato_valor(v):
    return f"{v:.4g}" if abs(v) < 1000 else f"{v:,.0f}"


def dotplot_variantes(series, xlabel, titulo, ruta_salida):
    """series: lista de (nombre, valores) -- una fila por serie, mismo eje X."""
    fig, ax = plt.subplots(figsize=(8.5, 0.9 * len(series) + 1.0))
    rng = np.random.default_rng(0)
    for i, (nombre, valores) in enumerate(series):
        valores = np.array(valores)
        y = len(series) - 1 - i
        y_jitter = y + rng.uniform(-0.15, 0.15, size=len(valores))
        color = COLORES_VARIANTES.get(nombre.split(" ")[0], "#4C72B0")
        ax.scatter(valores, y_jitter, color=color, edgecolors="black", s=60, zorder=3, alpha=0.85,
                   linewidths=0.8, label=nombre)
        media = valores.mean()
        ax.plot([media, media], [y - 0.28, y + 0.28], color="#333333", linestyle="--", linewidth=1.3, zorder=2)
        ax.text(media, y + 0.34, f"media = {_formato_valor(media)}", ha="center", fontsize=8, color="#333333")
    ax.set_yticks(range(len(series)))
    ax.set_yticklabels([s[0] for s in reversed(series)])
    ax.set_xlabel(xlabel)
    ax.set_title(titulo, fontweight="bold", fontsize=11)
    ax.grid(True, axis="x", linestyle="--", alpha=0.4)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    plt.tight_layout()
    Path(ruta_salida).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(ruta_salida, dpi=150)
    plt.close()


def main():
    for unidad in UNIDADES:
        carpeta = ROOT / "results_sgd_vs_adam" / unidad

        ruta_actual = carpeta / "metrics_seed_sweep.json"
        if ruta_actual.exists():
            datos = json.loads(ruta_actual.read_text(encoding="utf-8"))
            series = [(nombre, datos[nombre]["valores"]) for nombre in ("sgd", "adam") if nombre in datos]
            dotplot_variantes(series, "Accuracy en test", f"{unidad}: SGD vs Adam (N={series[0][1].__len__()} semillas)",
                               carpeta / "seed_sweep.png")
            print(f"[ok] {carpeta / 'seed_sweep.png'}")
        else:
            print(f"[saltado] {ruta_actual} no existe todavia")

        ruta_esquemaB = carpeta / "metrics_seed_sweep_esquemaB.json"
        if ruta_actual.exists() and ruta_esquemaB.exists():
            datos_actual = json.loads(ruta_actual.read_text(encoding="utf-8"))
            datos_esquemaB = json.loads(ruta_esquemaB.read_text(encoding="utf-8"))
            series = []
            for nombre in ("sgd", "adam"):
                if nombre in datos_actual:
                    series.append((f"{nombre} (split libre)", datos_actual[nombre]["valores"]))
                if nombre in datos_esquemaB:
                    series.append((f"{nombre} (split fijo)", datos_esquemaB[nombre]["valores"]))
            dotplot_variantes(series, "Accuracy en test", f"{unidad}: split libre vs split fijo",
                               carpeta / "seed_sweep_esquemaB.png")
            print(f"[ok] {carpeta / 'seed_sweep_esquemaB.png'}")
        else:
            print(f"[saltado] esquemaB de {unidad}: falta algun metrics_seed_sweep*.json")


if __name__ == "__main__":
    main()
