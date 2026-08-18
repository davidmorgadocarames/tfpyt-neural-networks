"""
Calibracion: curva de fiabilidad (confianza predicha vs accuracy real, por bins) sobre un
clasificador denso (Adam) entrenado sobre la muestra reducida de MNIST de
../1-sgd-vs-adam/sgd_vs_adam_dense_tf.py -- responde a "cuando el modelo dice que esta 90%
seguro, acierta realmente el 90% de las veces?", una pregunta distinta de la accuracy global y
relevante quando la confianza del modelo se usa para decidir (p. ej. escalar a revision humana
las predicciones de baja confianza).

Uso: python calibracion.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from _common import entrenar_dense_mnist

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

SEED_SPLIT = 42
SEED_MODELO = 42
N_BINS = 10


def main(quiet=False):
    model, x_test, y_test = entrenar_dense_mnist(SEED_SPLIT, SEED_MODELO, quiet=quiet)
    probs = model.predict(x_test, verbose=0)
    pred = np.argmax(probs, axis=1)
    confianza = np.max(probs, axis=1)
    aciertos = (pred == y_test).astype(float)

    bordes = np.linspace(0.0, 1.0, N_BINS + 1)
    confianza_media_bin, accuracy_bin, n_por_bin = [], [], []
    for i in range(N_BINS):
        lo, hi = bordes[i], bordes[i + 1]
        en_bin = (confianza > lo) & (confianza <= hi) if i > 0 else (confianza >= lo) & (confianza <= hi)
        n = int(en_bin.sum())
        n_por_bin.append(n)
        confianza_media_bin.append(float(confianza[en_bin].mean()) if n > 0 else None)
        accuracy_bin.append(float(aciertos[en_bin].mean()) if n > 0 else None)

    n_total = len(y_test)
    ece = sum(
        (n_por_bin[i] / n_total) * abs(accuracy_bin[i] - confianza_media_bin[i])
        for i in range(N_BINS) if n_por_bin[i] > 0
    )

    accuracy_global = float(aciertos.mean())
    confianza_media_global = float(confianza.mean())

    metrics = {
        "seed_split": SEED_SPLIT,
        "seed_modelo": SEED_MODELO,
        "n_bins": N_BINS,
        "bordes_bin": bordes.tolist(),
        "confianza_media_bin": confianza_media_bin,
        "accuracy_bin": accuracy_bin,
        "n_por_bin": n_por_bin,
        "ece": ece,
        "accuracy_global": accuracy_global,
        "confianza_media_global": confianza_media_global,
        "n_test": n_total,
    }
    (RESULTS_DIR / "calibracion_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    plt.figure(figsize=(5.5, 5.5))
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--", label="Calibracion perfecta")
    xs = [c for c in confianza_media_bin if c is not None]
    ys = [accuracy_bin[i] for i, c in enumerate(confianza_media_bin) if c is not None]
    plt.plot(xs, ys, marker="o", color="#C44E52", label="Modelo (curva de fiabilidad)")
    plt.xlabel("Confianza predicha (media por bin)")
    plt.ylabel("Accuracy real (por bin)")
    plt.title(f"Curva de fiabilidad -- ECE={ece:.4f}", fontweight="bold")
    plt.legend(fontsize=9)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "calibracion_reliability.png", dpi=150)
    plt.close()

    if not quiet:
        print(f"Accuracy global: {accuracy_global:.4f} | Confianza media: {confianza_media_global:.4f} | "
              f"ECE: {ece:.4f}")
        print(f"Resultados guardados en {RESULTS_DIR}")
    return metrics


if __name__ == "__main__":
    main()
