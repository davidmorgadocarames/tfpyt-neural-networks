"""
SGD vs Adam (Keras) sobre una muestra reducida de MNIST, full-batch (batch_size = todo el
train, 1 actualizacion de pesos por epoca) -- equivalente TF de
numpy-neural-networks-from-scratch/07-reconocimiento-digitos/sgd_vs_adam.py.

Misma arquitectura que 01_dense_classifier_mnist.py (no la reimplementa: build_model() vive
aqui parametrizada por optimizador, igual que crear_red() en el script RRNN de referencia).
No toca 01_dense_classifier_mnist.py -- guarda sus propios resultados en
results_sgd_vs_adam/dense_tf_fullbatch/.

Muestra: 1200 train / 300 val / 300 test, 120/30/30 por digito (mismo tamano que RRNN), para
que "full-batch" sea barato de verdad. seed_split decide que imagenes caen en cada split;
seed_modelo (via tf.keras.utils.set_random_seed) decide inicializacion de pesos y mascaras de
Dropout -- misma separacion que en RRNN.

Uso: python sgd_vs_adam_dense_tf.py
"""

import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

RESULTS_DIR = Path(__file__).parent / "results_sgd_vs_adam" / "dense_tf_fullbatch"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

N_TRAIN, N_VAL, N_TEST = 1200, 300, 300  # 120/30/30 por digito, igual que RRNN 07
EPOCHS_MAX = 600
PATIENCE = 60  # ~10% de EPOCHS_MAX -- full-batch converge muy despacio (1 update/epoca)

VARIANTES = [
    ("sgd", lambda: tf.keras.optimizers.SGD(learning_rate=0.5, momentum=0.0)),
    ("adam", lambda: tf.keras.optimizers.Adam(learning_rate=0.005)),
]
COLORES_VARIANTES = {"sgd": "#4C72B0", "adam": "#55A868"}


def cargar_datos(seed_split, quiet=False):
    """Junta train+test de Keras (70.000 imagenes) y muestrea de forma estratificada por
    digito, igual que digit_classifier.py en RRNN -- mismos tamanos (120/30/30 por clase)."""
    (x_tr, y_tr), (x_te, y_te) = tf.keras.datasets.mnist.load_data()
    X = np.concatenate([x_tr, x_te], axis=0).reshape(-1, 28 * 28).astype("float32") / 255.0
    Y = np.concatenate([y_tr, y_te], axis=0).astype("int64")

    rng = np.random.default_rng(seed_split)
    idx_train, idx_val, idx_test = [], [], []
    for digito in range(10):
        idx_digito = np.where(Y == digito)[0]
        rng.shuffle(idx_digito)
        c_train = N_TRAIN // 10
        c_val = c_train + N_VAL // 10
        c_test = c_val + N_TEST // 10
        idx_train.extend(idx_digito[:c_train])
        idx_val.extend(idx_digito[c_train:c_val])
        idx_test.extend(idx_digito[c_val:c_test])
    idx_train, idx_val, idx_test = np.array(idx_train), np.array(idx_val), np.array(idx_test)
    rng.shuffle(idx_train)
    rng.shuffle(idx_val)
    rng.shuffle(idx_test)

    if not quiet:
        print(f"MNIST muestreado: {len(idx_train)} train / {len(idx_val)} val / {len(idx_test)} test")
    return X[idx_train], Y[idx_train], X[idx_val], Y[idx_val], X[idx_test], Y[idx_test]


def build_model(optimizer) -> tf.keras.Model:
    """Misma arquitectura que 01_dense_classifier_mnist.py (784 -> 128 LeakyReLU -> Dropout ->
    64 LeakyReLU -> Dropout -> 10 softmax) -- solo cambia el optimizador."""
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(28 * 28,)),
            tf.keras.layers.Dense(128, activation=tf.keras.layers.LeakyReLU(negative_slope=0.01)),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(64, activation=tf.keras.layers.LeakyReLU(negative_slope=0.01)),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(10, activation="softmax"),
        ]
    )
    model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


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


def graficar_curva_comparativa(curvas_val, ruta_salida):
    plt.figure(figsize=(8, 4.5))
    for nombre, historial in curvas_val.items():
        plt.plot(historial, color=COLORES_VARIANTES[nombre], label=nombre.upper())
    plt.yscale("log")
    plt.title("SGD vs Adam (Keras): perdida de validacion por epoca (muestra reducida, full-batch)",
               fontweight="bold")
    plt.xlabel("Epoca")
    plt.ylabel("Perdida de validacion (escala log)")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(ruta_salida, dpi=150)
    plt.close()


def main(seed_split=42, seed_modelo=42, quiet=False, guardar_graficas=True) -> dict:
    x_train, y_train, x_val, y_val, x_test, y_test = cargar_datos(seed_split, quiet=quiet)

    resultados = {}
    curvas_val = {}
    for nombre, crear_optimizador in VARIANTES:
        if not quiet:
            print(f"\n=== Entrenando '{nombre}' (Keras, full-batch) ===")
        tf.keras.utils.set_random_seed(seed_modelo)
        tf.config.experimental.enable_op_determinism()

        model = build_model(crear_optimizador())
        t0 = time.time()
        history = model.fit(
            x_train, y_train,
            validation_data=(x_val, y_val),
            epochs=EPOCHS_MAX,
            batch_size=len(x_train),  # full-batch: 1 actualizacion de pesos por epoca
            callbacks=[tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=PATIENCE, restore_best_weights=True)],
            verbose=2 if not quiet else 0,
        )
        tiempo = time.time() - t0

        val_loss_hist = history.history["val_loss"]
        epoca_mejor = int(np.argmin(val_loss_hist))
        epocas_entrenadas = len(val_loss_hist)

        test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
        y_pred = np.argmax(model.predict(x_test, verbose=0), axis=1)
        matriz_confusion = np.zeros((10, 10), dtype=int)
        for real, pred in zip(y_test, y_pred):
            matriz_confusion[real, pred] += 1

        if not quiet:
            print(f"  [{nombre}] accuracy test: {test_acc:.4f} -- epocas: {epocas_entrenadas} "
                  f"(mejor val en {epoca_mejor + 1}) -- {tiempo:.1f}s")

        if guardar_graficas:
            graficar_confusion(matriz_confusion, test_acc, f"Matriz de confusion test -- {nombre.upper()}",
                                RESULTS_DIR / f"confusion_matrix_{nombre}.png")

        resultados[nombre] = {
            "epochs_entrenadas": epocas_entrenadas,
            "epoca_mejor_val": epoca_mejor + 1,
            "loss_val_final": float(val_loss_hist[epoca_mejor]),
            "accuracy_test": float(test_acc),
            "loss_test": float(test_loss),
            "tiempo_segundos": tiempo,
            "matriz_confusion": matriz_confusion.tolist(),
            "historial_loss_val": val_loss_hist,
        }
        curvas_val[nombre] = val_loss_hist

    metrics = {
        "seed_split": seed_split,
        "seed_modelo": seed_modelo,
        "epochs_max_configuradas": EPOCHS_MAX,
        "n_train": len(x_train), "n_val": len(x_val), "n_test": len(x_test),
        "resultados": resultados,
    }

    if not guardar_graficas:
        return metrics

    (RESULTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    graficar_curva_comparativa(curvas_val, RESULTS_DIR / "learning_curve_comparativa.png")

    if not quiet:
        print("\n--- Resumen ---")
        for nombre, datos in resultados.items():
            print(f"{nombre:6s} accuracy={datos['accuracy_test']:.4f}  epocas={datos['epochs_entrenadas']}  "
                  f"tiempo={datos['tiempo_segundos']:.1f}s")
        print(f"Resultados guardados en {RESULTS_DIR}")

    return metrics


if __name__ == "__main__":
    main()
