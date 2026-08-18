"""
Validacion cruzada k-fold (k=5, estratificada) sobre el problema de tipos de clientes --
alternativa metodologica al barrido de semillas Monte Carlo (run_seed_sweep.py en
../1-sgd-vs-adam/): en vez de repetir N veces con splits aleatorios independientes (que pueden
solaparse entre si), se parte el dataset completo (120 clientes) en 5 folds disjuntos y cada
uno sirve de test exactamente una vez, usando el resto como train -- la varianza se mide sobre
particiones que cubren el 100% de los datos exactamente una vez cada una, no sobre muestreos
aleatorios independientes.

Reutiliza generar_datos() y build_model() de sgd_vs_adam_customer_tf.py -- no reimplementa la
arquitectura ni la generacion de datos.

Uso: python kfold_cross_validation.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold

from _common import cargar_modulo

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

K_FOLDS = 5
SEED_KFOLD = 42  # gobierna solo el reparto en folds -- los datos siempre usan SEED_DATOS (fijo
# dentro de sgd_vs_adam_customer_tf.py, nunca cambia)
SEED_MODELO = 42
EPOCHS = 800  # fijo, sin conjunto de validacion propio por fold (ver nota en el bucle)


def main(quiet=False):
    customer_tf = cargar_modulo("sgd_vs_adam_customer_tf.py")
    X, Y_num = customer_tf.generar_datos()

    skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=SEED_KFOLD)
    accuracies = []
    historiales_loss_train = []

    for fold, (idx_train, idx_test) in enumerate(skf.split(X, Y_num)):
        X_min, X_max = X[idx_train].min(axis=0), X[idx_train].max(axis=0)
        X_train = ((X[idx_train] - X_min) / (X_max - X_min)).astype("float32")
        X_test = ((X[idx_test] - X_min) / (X_max - X_min)).astype("float32")
        Y_train, Y_test = Y_num[idx_train], Y_num[idx_test]

        tf.keras.utils.set_random_seed(SEED_MODELO)
        tf.config.experimental.enable_op_determinism()
        model = customer_tf.build_model(tf.keras.optimizers.Adam(learning_rate=0.02))
        # Sin EarlyStopping/validacion propia dentro de cada fold: con solo 96 muestras de
        # train por fold, partir ademas una validacion dejaria muy pocas muestras por clase.
        # EPOCHS=800 esta muy por encima de lo que necesita este problema para converger (ver
        # sgd_vs_adam_customer_tf.py, que con Adam converge en menos de 2000 epocas sobre un
        # conjunto de train mas pequeno todavia).
        history = model.fit(X_train, Y_train, epochs=EPOCHS, batch_size=len(X_train), verbose=0)

        loss_test, acc_test = model.evaluate(X_test, Y_test, verbose=0)
        accuracies.append(float(acc_test))
        historiales_loss_train.append(history.history["loss"])
        if not quiet:
            print(f"Fold {fold + 1}/{K_FOLDS}: accuracy test = {acc_test:.4f} ({len(idx_test)} muestras)")

    media = float(np.mean(accuracies))
    desviacion = float(np.std(accuracies))

    metrics = {
        "k_folds": K_FOLDS,
        "seed_kfold": SEED_KFOLD,
        "seed_modelo": SEED_MODELO,
        "epochs": EPOCHS,
        "accuracies_por_fold": accuracies,
        "media": media,
        "desviacion_tipica": desviacion,
        "historiales_loss_train": historiales_loss_train,
    }
    (RESULTS_DIR / "kfold_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    plt.figure(figsize=(6, 4.5))
    plt.bar(range(1, K_FOLDS + 1), accuracies, color="#4C72B0")
    plt.axhline(media, color="#C44E52", linestyle="--",
                label=f"Media ({media:.4f} +/- {desviacion:.4f})")
    plt.xlabel("Fold")
    plt.ylabel("Accuracy test")
    plt.title(f"Validacion cruzada k-fold (k={K_FOLDS}) -- tipos de clientes", fontweight="bold")
    plt.ylim(0, 1.05)
    plt.legend()
    plt.grid(True, axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "kfold_accuracy.png", dpi=150)
    plt.close()

    if not quiet:
        print(f"\nAccuracy media: {media:.4f} +/- {desviacion:.4f}")
        print(f"Resultados guardados en {RESULTS_DIR}")
    return metrics


if __name__ == "__main__":
    main()
