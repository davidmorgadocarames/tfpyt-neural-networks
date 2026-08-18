"""
Ensembling: promedia las predicciones (softmax) de K clasificadores densos entrenados de forma
independiente -- mismo split de datos (seed_split=42), solo cambia seed_modelo -- sobre la
muestra reducida de MNIST de ../1-sgd-vs-adam/sgd_vs_adam_dense_tf.py, y compara la accuracy
del ensemble frente a la de cada modelo individual. Es la pregunta metodologica clasica de si
promediar N modelos "razonables pero imperfectos" reduce el error frente a quedarse con uno
solo -- relevante en produccion porque no hace falta encontrar "la" mejor semilla, con votar
varias basta.

Uso: python ensembling.py [--k 7]
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from _common import entrenar_dense_mnist

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

SEED_SPLIT = 42  # fijo: el ensemble aisla el efecto de re-inicializar pesos, no el del split
K_DEFAULT = 7


def main(k=K_DEFAULT, quiet=False):
    probs_por_modelo = []
    accuracies_individuales = []
    y_test_ref = None

    for i in range(k):
        seed_modelo = 1000 + i
        if not quiet:
            print(f"Entrenando modelo {i + 1}/{k} (seed_modelo={seed_modelo})...")
        model, x_test, y_test = entrenar_dense_mnist(SEED_SPLIT, seed_modelo, quiet=True)
        if y_test_ref is None:
            y_test_ref = y_test
        probs = model.predict(x_test, verbose=0)
        probs_por_modelo.append(probs)
        acc = float(np.mean(np.argmax(probs, axis=1) == y_test))
        accuracies_individuales.append(acc)
        if not quiet:
            print(f"  accuracy individual: {acc:.4f}")

    probs_ensemble = np.mean(probs_por_modelo, axis=0)
    pred_ensemble = np.argmax(probs_ensemble, axis=1)
    accuracy_ensemble = float(np.mean(pred_ensemble == y_test_ref))

    media_individual = float(np.mean(accuracies_individuales))
    mejor_individual = float(np.max(accuracies_individuales))
    peor_individual = float(np.min(accuracies_individuales))

    metrics = {
        "k": k,
        "seed_split": SEED_SPLIT,
        "accuracies_individuales": accuracies_individuales,
        "media_individual": media_individual,
        "mejor_individual": mejor_individual,
        "peor_individual": peor_individual,
        "accuracy_ensemble": accuracy_ensemble,
        "n_test": int(len(y_test_ref)),
    }
    (RESULTS_DIR / "ensembling_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    plt.figure(figsize=(7, 4.5))
    xs = list(range(1, k + 1))
    plt.bar(xs, accuracies_individuales, color="#4C72B0", label="Modelos individuales")
    plt.axhline(media_individual, color="#4C72B0", linestyle=":",
                label=f"Media individual ({media_individual:.4f})")
    plt.axhline(accuracy_ensemble, color="#C44E52", linestyle="--", linewidth=2,
                label=f"Ensemble ({accuracy_ensemble:.4f})")
    plt.xlabel("Modelo")
    plt.ylabel("Accuracy test")
    plt.title(f"Ensembling: {k} clasificadores densos (MNIST reducido) vs su promedio",
              fontweight="bold")
    plt.legend(fontsize=8)
    plt.grid(True, axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "ensembling_accuracy.png", dpi=150)
    plt.close()

    if not quiet:
        print(f"\nMedia individual: {media_individual:.4f} | Mejor individual: {mejor_individual:.4f} | "
              f"Ensemble: {accuracy_ensemble:.4f}")
        print(f"Resultados guardados en {RESULTS_DIR}")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=K_DEFAULT)
    args = parser.parse_args()
    main(k=args.k)
