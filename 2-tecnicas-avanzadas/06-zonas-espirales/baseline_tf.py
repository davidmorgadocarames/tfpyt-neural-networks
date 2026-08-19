"""
Baseline (Keras) del problema de zonas de espirales -- el script "main" de este problema.
Importa las constantes de adam.py/lr_decay.py/weight_decay.py y monta UN optimizador AdamW con
LR-schedule y weight decay ya combinados -- ver ../03-tipos-clientes/baseline_tf.py para la
explicacion completa del diseno.

Mismos datos que el problema 06 de RRNN: 450 puntos (150/brazo, SEED_DATOS=0), split 60/20/20
estratificado por brazo. Red 2 -> 64 (LeakyReLU) -> 64 (LeakyReLU) -> 3 (softmax), full-batch.

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
from lr_decay import DECAY_EVERY, DECAY_RATE
from weight_decay import WEIGHT_DECAY

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

SEED_DATOS = 0
N_POR_BRAZO = 150
K_CLASES = 3
EPOCHS_MAX = 5000
PATIENCE = 200
MIN_DELTA = 1e-4


def generar_datos():
    """Identico al generador de RRNN/06-zonas-espirales, SIEMPRE con SEED_DATOS."""
    np.random.seed(SEED_DATOS)
    X = np.zeros((N_POR_BRAZO * K_CLASES, 2), dtype="float32")
    Y_num = np.zeros(N_POR_BRAZO * K_CLASES, dtype="int64")
    for i in range(K_CLASES):
        r = np.linspace(0.0, 1, N_POR_BRAZO)
        t = np.linspace(i * 4, (i + 1) * 4, N_POR_BRAZO) + np.random.randn(N_POR_BRAZO) * 0.2
        X[i * N_POR_BRAZO : (i + 1) * N_POR_BRAZO] = np.c_[r * np.sin(t), r * np.cos(t)]
        Y_num[i * N_POR_BRAZO : (i + 1) * N_POR_BRAZO] = i
    return X, Y_num


def split_estratificado(Y_num, seed_split):
    rng = np.random.default_rng(seed_split)
    indices_train, indices_val, indices_test = [], [], []
    for clase in range(K_CLASES):
        idx_clase = np.where(Y_num == clase)[0]
        rng.shuffle(idx_clase)
        corte_train = int(0.6 * len(idx_clase))
        corte_val = int(0.8 * len(idx_clase))
        indices_train.extend(idx_clase[:corte_train])
        indices_val.extend(idx_clase[corte_train:corte_val])
        indices_test.extend(idx_clase[corte_val:])
    return np.array(indices_train), np.array(indices_val), np.array(indices_test)


def cargar_datos(seed_split, quiet=False):
    X, Y_num = generar_datos()
    indices_train, indices_val, indices_test = split_estratificado(Y_num, seed_split)
    X_train, X_val, X_test = X[indices_train], X[indices_val], X[indices_test]
    Y_train, Y_val, Y_test = Y_num[indices_train], Y_num[indices_val], Y_num[indices_test]
    if not quiet:
        print(f"Espirales: {len(X_train)} train / {len(X_val)} val / {len(X_test)} test")
    return X_train, Y_train, X_val, Y_val, X_test, Y_test


def build_model() -> tf.keras.Model:
    return tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(2,)),
            tf.keras.layers.Dense(64, activation=tf.keras.layers.LeakyReLU(negative_slope=0.01)),
            tf.keras.layers.Dense(64, activation=tf.keras.layers.LeakyReLU(negative_slope=0.01)),
            tf.keras.layers.Dense(3, activation="softmax"),
        ]
    )


def crear_optimizador() -> tf.keras.optimizers.Optimizer:
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=LEARNING_RATE_ADAM, decay_steps=DECAY_EVERY, decay_rate=DECAY_RATE, staircase=True)
    return tf.keras.optimizers.AdamW(learning_rate=lr_schedule, weight_decay=WEIGHT_DECAY)


def graficar_confusion(matriz, accuracy, titulo, ruta):
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(matriz, cmap="Blues")
    ax.set_title(f"{titulo} (accuracy={accuracy:.2%})", fontweight="bold")
    ax.set_xlabel("Prediccion")
    ax.set_ylabel("Real")
    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels(["Brazo 0", "Brazo 1", "Brazo 2"])
    ax.set_yticklabels(["Brazo 0", "Brazo 1", "Brazo 2"])
    for i in range(3):
        for j in range(3):
            ax.text(j, i, str(matriz[i, j]), ha="center", va="center",
                     color="white" if matriz[i, j] > matriz.max() / 2 else "black", fontsize=11)
    fig.colorbar(im, fraction=0.046)
    plt.tight_layout()
    plt.savefig(ruta, dpi=150)
    plt.close()


def graficar_curva(historial_loss_val, ruta_salida):
    plt.figure(figsize=(7, 4.5))
    plt.plot(historial_loss_val, color="#4C72B0")
    plt.yscale("log")
    plt.title("Baseline (Adam+LRdecay+WeightDecay): perdida de validacion por epoca", fontweight="bold")
    plt.xlabel("Epoca")
    plt.ylabel("Perdida de validacion (escala log)")
    plt.grid(True, which="both", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(ruta_salida, dpi=150)
    plt.close()


def main(seed_split=0, seed_modelo=0, quiet=False, guardar_graficas=True) -> dict:
    x_train, y_train, x_val, y_val, x_test, y_test = cargar_datos(seed_split, quiet=quiet)

    tf.keras.utils.set_random_seed(seed_modelo)
    tf.config.experimental.enable_op_determinism()

    model = build_model()
    model.compile(optimizer=crear_optimizador(), loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    t0 = time.time()
    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=EPOCHS_MAX,
        batch_size=len(x_train),
        callbacks=[tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=PATIENCE, min_delta=MIN_DELTA, restore_best_weights=True)],
        verbose=2 if not quiet else 0,
    )
    tiempo_computacion_segundos = time.time() - t0

    val_loss_hist = history.history["val_loss"]
    epoca_mejor = int(np.argmin(val_loss_hist))
    epocas_entrenadas = len(val_loss_hist)

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    y_pred = np.argmax(model.predict(x_test, verbose=0), axis=1)
    matriz_confusion = np.zeros((3, 3), dtype=int)
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
        "decay_every": DECAY_EVERY,
        "weight_decay": WEIGHT_DECAY,
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
