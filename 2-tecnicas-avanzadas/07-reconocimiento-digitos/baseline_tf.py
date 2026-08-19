"""
Baseline (Keras) del reconocimiento de digitos MNIST -- el script "main" de este problema.
Importa las constantes de adam.py/lr_decay.py/weight_decay.py y monta UN optimizador AdamW con
LR-schedule y weight decay ya combinados -- ver ../03-tipos-clientes/baseline_tf.py para la
explicacion completa del diseno.

A diferencia de 03/06/04 (full-batch, redes pequenas), este es uno de "las redes
convencionales" del apartado: MNIST completo (60k/10k), mini-batch (batch_size=128), y
BatchNorm integrado en la arquitectura -- Dense(128) -> BatchNorm -> LeakyReLU ->
Dense(10, softmax), el mismo orden Dense->BatchNorm->activacion del paper de Ioffe & Szegedy
(2015) que usa RRNN/07-reconocimiento-digitos/batchnorm.py.

Uso: python baseline_tf.py
"""

import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from adam import LEARNING_RATE_ADAM
from lr_decay import DECAY_RATE
from weight_decay import WEIGHT_DECAY

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

BATCH_SIZE = 128
EPOCHS_MAX = 30
PATIENCE = 5


def cargar_datos(seed_split, quiet=False):
    """Split 90/10 de train (para reservar validacion propia) + el test estandar de MNIST."""
    (x_train_full, y_train_full), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train_full = x_train_full.reshape(-1, 28 * 28).astype("float32") / 255.0
    x_test = x_test.reshape(-1, 28 * 28).astype("float32") / 255.0
    y_train_full = y_train_full.astype("int64")
    y_test = y_test.astype("int64")

    rng = np.random.default_rng(seed_split)
    n = len(x_train_full)
    indices = rng.permutation(n)
    corte = int(n * 0.9)
    idx_train, idx_val = indices[:corte], indices[corte:]

    if not quiet:
        print(f"MNIST completo: {len(idx_train)} train / {len(idx_val)} val / {len(x_test)} test")
    return (x_train_full[idx_train], y_train_full[idx_train],
            x_train_full[idx_val], y_train_full[idx_val], x_test, y_test)


def build_model() -> tf.keras.Model:
    """Dense(128) -> BatchNorm -> LeakyReLU -> Dense(10, softmax).

    BatchNormalization usa momentum=0.9 (en vez del 0.99 por defecto de Keras) -- verificado con
    ensembling_kfold_tf.py (solo 30 steps/epoca sobre la muestra reducida) que con el momentum
    por defecto la media movil de la normalizacion tarda demasiados steps en converger (0.99**30
    todavia conserva un 74% del estado inicial), desincronizando estadisticas de entrenamiento
    vs inferencia. momentum=0.9 iguala el comportamiento por defecto de PyTorch (que usa la
    convencion opuesta: su momentum=0.1 pondera igual que un momentum=0.9 aqui), y mantiene la
    paridad entre ambos frameworks."""
    return tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(28 * 28,)),
            tf.keras.layers.Dense(128),
            tf.keras.layers.BatchNormalization(momentum=0.9),
            tf.keras.layers.LeakyReLU(negative_slope=0.01),
            tf.keras.layers.Dense(10, activation="softmax"),
        ]
    )


def crear_optimizador(steps_per_epoch) -> tf.keras.optimizers.Optimizer:
    """Adam + LR decay + weight decay combinados. steps_per_epoch (no 1): en Keras,
    decay_steps de ExponentialDecay cuenta pasos de optimizador (mini-batches), no epocas."""
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=LEARNING_RATE_ADAM, decay_steps=steps_per_epoch, decay_rate=DECAY_RATE, staircase=True)
    return tf.keras.optimizers.AdamW(learning_rate=lr_schedule, weight_decay=WEIGHT_DECAY)


def graficar_confusion(matriz, accuracy, titulo, ruta):
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(matriz, cmap="Blues")
    ax.set_title(f"{titulo} (accuracy={accuracy:.2%})", fontweight="bold")
    ax.set_xlabel("Prediccion")
    ax.set_ylabel("Real")
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    for i in range(10):
        for j in range(10):
            valor = matriz[i, j]
            if valor > 0:
                ax.text(j, i, str(valor), ha="center", va="center",
                         color="white" if valor > matriz.max() / 2 else "black", fontsize=8)
    fig.colorbar(im, fraction=0.046)
    plt.tight_layout()
    plt.savefig(ruta, dpi=150)
    plt.close()


def graficar_curva(historial_loss_val, ruta_salida):
    plt.figure(figsize=(7, 4.5))
    plt.plot(historial_loss_val, color="#4C72B0")
    plt.yscale("log")
    plt.title("Baseline (Adam+LRdecay+WeightDecay+BatchNorm): perdida de validacion", fontweight="bold")
    plt.xlabel("Epoca")
    plt.ylabel("Perdida de validacion (escala log)")
    plt.grid(True, which="both", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(ruta_salida, dpi=150)
    plt.close()


def main(seed_split=42, seed_modelo=42, quiet=False, guardar_graficas=True) -> dict:
    x_train, y_train, x_val, y_val, x_test, y_test = cargar_datos(seed_split, quiet=quiet)
    steps_per_epoch = -(-len(x_train) // BATCH_SIZE)  # ceil division

    tf.keras.utils.set_random_seed(seed_modelo)
    tf.config.experimental.enable_op_determinism()

    model = build_model()
    model.compile(optimizer=crear_optimizador(steps_per_epoch), loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    t0 = time.time()
    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=EPOCHS_MAX,
        batch_size=BATCH_SIZE,
        callbacks=[tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=PATIENCE, restore_best_weights=True)],
        verbose=2 if not quiet else 0,
    )
    tiempo_computacion_segundos = time.time() - t0

    val_loss_hist = history.history["val_loss"]
    epoca_mejor = int(np.argmin(val_loss_hist))
    epocas_entrenadas = len(val_loss_hist)

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    y_pred = np.argmax(model.predict(x_test, verbose=0), axis=1)
    matriz_confusion = np.zeros((10, 10), dtype=int)
    for real, pred in zip(y_test, y_pred):
        matriz_confusion[real, pred] += 1

    if not quiet:
        print(f"accuracy test: {test_acc:.4f} -- epocas: {epocas_entrenadas} "
              f"(mejor val en {epoca_mejor + 1}) -- {tiempo_computacion_segundos:.1f}s")

    metrics = {
        "seed_split": seed_split,
        "seed_modelo": seed_modelo,
        "learning_rate_adam": LEARNING_RATE_ADAM,
        "decay_rate": DECAY_RATE,
        "steps_per_epoch": steps_per_epoch,
        "weight_decay": WEIGHT_DECAY,
        "batch_size": BATCH_SIZE,
        "epochs_max_configuradas": EPOCHS_MAX,
        "epochs_entrenadas": epocas_entrenadas,
        "epoca_mejor_val": epoca_mejor + 1,
        "loss_val_final": float(val_loss_hist[epoca_mejor]),
        "accuracy_test": float(test_acc),
        "loss_test": float(test_loss),
        "tiempo_computacion_segundos": tiempo_computacion_segundos,
        "n_train": len(x_train), "n_val": len(x_val), "n_test": len(x_test),
        "matriz_confusion": matriz_confusion.tolist(),
        "historial_loss_val": val_loss_hist,
    }

    if not guardar_graficas:
        return metrics

    (RESULTS_DIR / "baseline_tf_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    graficar_confusion(matriz_confusion, test_acc, "Matriz de confusion test -- baseline",
                        RESULTS_DIR / "baseline_tf_confusion.png")
    graficar_curva(val_loss_hist, RESULTS_DIR / "baseline_tf_curva.png")

    if not quiet:
        print(f"Resultados guardados en {RESULTS_DIR}")

    return metrics


if __name__ == "__main__":
    main()
