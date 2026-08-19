"""
Baseline (Keras) del problema de tipos de clientes -- el script "main" de este problema.
Importa las constantes de adam.py/lr_decay.py/weight_decay.py y monta UN optimizador AdamW
con learning-rate-schedule (decaimiento exponencial) y weight decay ya combinados -- Adam +
LR decay + weight decay se aplican todos juntos a la misma red, en una unica configuracion,
no como comparaciones aisladas "con vs sin". Es el punto de partida que
ensembling_kfold_tf.py importa.

Mismos datos que el problema 03 de RRNN: 120 clientes (40/categoria, SEED_DATOS=42), split
60/20/20 estratificado, normalizacion min-max con min/max de train. Red 2 -> 5 (LeakyReLU) -> 3
(softmax), full-batch.

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

SEED_DATOS = 42  # gobierna solo la generacion de datos -- fijo siempre
N_TRAIN_CLASE, N_VAL_CLASE = 24, 8  # 24 train / 8 val / 8 test por categoria (60/20/20)
NOMBRES_CLASES = ["Navegadores", "Ocasionales", "VIPs"]
EPOCHS_MAX = 3000
PATIENCE = 200
MIN_DELTA = 1e-4  # sin Dropout, full-batch, determinista: hace falta min_delta para que el
# plateau dispare el early stopping (ver ../../1-sgd-vs-adam/sgd_vs_adam_customer_tf.py)


def generar_datos():
    """120 clientes, 40 por categoria, SIEMPRE con SEED_DATOS."""
    np.random.seed(SEED_DATOS)
    X0 = np.random.uniform(2, 10, (40, 2)) + np.array([0, 0])
    X1 = np.random.uniform(15, 25, (40, 2)) + np.array([0, 2])
    X2 = np.random.uniform(30, 45, (40, 2)) + np.array([0, 6])
    X = np.vstack([X0, X1, X2])
    Y_num = np.concatenate([np.zeros(40), np.ones(40), np.full(40, 2)]).astype("int64")
    return X, Y_num


def cargar_datos(seed_split, quiet=False):
    """Split train/val/test estratificado (60/20/20) + normalizacion min-max con min/max de
    train."""
    X, Y_num = generar_datos()

    rng = np.random.default_rng(seed_split)
    indices_train, indices_val, indices_test = [], [], []
    for clase in range(3):
        idx_clase = np.where(Y_num == clase)[0]
        rng.shuffle(idx_clase)
        indices_train.extend(idx_clase[:N_TRAIN_CLASE])
        indices_val.extend(idx_clase[N_TRAIN_CLASE : N_TRAIN_CLASE + N_VAL_CLASE])
        indices_test.extend(idx_clase[N_TRAIN_CLASE + N_VAL_CLASE :])
    indices_train = np.array(indices_train)
    indices_val = np.array(indices_val)
    indices_test = np.array(indices_test)

    X_train_raw, X_val_raw, X_test_raw = X[indices_train], X[indices_val], X[indices_test]
    Y_train, Y_val, Y_test = Y_num[indices_train], Y_num[indices_val], Y_num[indices_test]

    X_min, X_max = X_train_raw.min(axis=0), X_train_raw.max(axis=0)
    X_train = ((X_train_raw - X_min) / (X_max - X_min)).astype("float32")
    X_val = ((X_val_raw - X_min) / (X_max - X_min)).astype("float32")
    X_test = ((X_test_raw - X_min) / (X_max - X_min)).astype("float32")

    if not quiet:
        print(f"Clientes: {len(X_train)} train / {len(X_val)} val / {len(X_test)} test")
    return X_train, Y_train, X_val, Y_val, X_test, Y_test


def build_model() -> tf.keras.Model:
    """2 -> 5 LeakyReLU -> 3 softmax, sin Dropout (red demasiado pequena para necesitar
    regularizar mas alla de weight decay)."""
    return tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(2,)),
            tf.keras.layers.Dense(5, activation=tf.keras.layers.LeakyReLU(negative_slope=0.01)),
            tf.keras.layers.Dense(3, activation="softmax"),
        ]
    )


def crear_optimizador() -> tf.keras.optimizers.Optimizer:
    """Adam + LR decay + weight decay combinados en un unico optimizador AdamW. AdamW (no
    Adam+kernel_regularizer=l2) porque aplica el decaimiento de pesos desacoplado del
    gradiente adaptativo (Loshchilov & Hutter, 2019), no L2 acoplado al gradiente."""
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
    ax.set_xticklabels(NOMBRES_CLASES, rotation=20, ha="right")
    ax.set_yticklabels(NOMBRES_CLASES)
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


def main(seed_split=42, seed_modelo=42, quiet=False, guardar_graficas=True) -> dict:
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
