"""
SGD vs Adam (Keras) sobre una muestra reducida de Fashion-MNIST, full-batch (batch_size = todo
el train, 1 actualizacion de pesos por epoca) -- equivalente TF de
numpy-neural-networks-from-scratch/08-cnn-fashion-mnist/sgd_vs_adam.py (aqui solo la variante
baseline, sin augmentation -- ese es un experimento distinto ya cubierto por 02_cnn_vision.py).

Misma arquitectura CNN que 02_cnn_vision.py con augment=False (Conv2D 32 -> MaxPool -> Conv2D
64 -> MaxPool -> Flatten -> Dropout -> Dense 128 -> Dense 10) -- no lo toca, guarda sus propios
resultados en results_sgd_vs_adam/cnn_tf_fullbatch/.

Muestra: 2400 train / 600 val / 1000 test, 240/60/100 por clase (mismo tamano que RRNN 08).

Uso: python sgd_vs_adam_cnn_tf.py
"""

import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

RESULTS_DIR = Path(__file__).parent / "results_sgd_vs_adam" / "cnn_tf_fullbatch"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = [
    "Camiseta", "Pantalon", "Jersey", "Vestido", "Abrigo",
    "Sandalia", "Camisa", "Zapatilla", "Bolso", "Bota",
]

N_TRAIN, N_VAL, N_TEST = 2400, 600, 1000  # 240/60/100 por clase, igual que RRNN 08
EPOCHS_MAX = 700
PATIENCE = 40

VARIANTES = [
    ("sgd", lambda: tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.0)),
    ("adam", lambda: tf.keras.optimizers.Adam(learning_rate=0.001)),
]
COLORES_VARIANTES = {"sgd": "#4C72B0", "adam": "#55A868"}


def cargar_datos(seed_split, quiet=False):
    """Junta train+test de Fashion-MNIST (70.000 imagenes) y muestrea de forma estratificada
    por clase, igual que cnn_fashion_mnist.py en RRNN -- mismos tamanos (240/60/100 por
    clase)."""
    (x_tr, y_tr), (x_te, y_te) = tf.keras.datasets.fashion_mnist.load_data()
    X = np.concatenate([x_tr, x_te], axis=0)[..., np.newaxis].astype("float32")
    Y = np.concatenate([y_tr, y_te], axis=0).astype("int64")

    rng = np.random.default_rng(seed_split)
    idx_train, idx_val, idx_test = [], [], []
    for clase in range(10):
        idx_clase = np.where(Y == clase)[0]
        rng.shuffle(idx_clase)
        c_train = N_TRAIN // 10
        c_val = c_train + N_VAL // 10
        c_test = c_val + N_TEST // 10
        idx_train.extend(idx_clase[:c_train])
        idx_val.extend(idx_clase[c_train:c_val])
        idx_test.extend(idx_clase[c_val:c_test])
    idx_train, idx_val, idx_test = np.array(idx_train), np.array(idx_val), np.array(idx_test)
    rng.shuffle(idx_train)
    rng.shuffle(idx_val)
    rng.shuffle(idx_test)

    if not quiet:
        print(f"Fashion-MNIST muestreado: {len(idx_train)} train / {len(idx_val)} val / {len(idx_test)} test")
    return X[idx_train], Y[idx_train], X[idx_val], Y[idx_val], X[idx_test], Y[idx_test]


def build_model(optimizer) -> tf.keras.Model:
    """Misma arquitectura que 02_cnn_vision.py con augment=False -- solo cambia el
    optimizador."""
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(28, 28, 1)),
            tf.keras.layers.Rescaling(1.0 / 255),
            tf.keras.layers.Conv2D(32, 3, activation="relu", padding="same"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(64, 3, activation="relu", padding="same"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(10, activation="softmax"),
        ]
    )
    model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def graficar_confusion(matriz, accuracy, titulo, ruta):
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(matriz, cmap="Blues")
    ax.set_title(f"{titulo} (accuracy={accuracy:.2%})", fontweight="bold")
    ax.set_xlabel("Prediccion")
    ax.set_ylabel("Real")
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.set_xticklabels(CLASS_NAMES, rotation=90, fontsize=7)
    ax.set_yticklabels(CLASS_NAMES, fontsize=7)
    for i in range(10):
        for j in range(10):
            valor = matriz[i, j]
            if valor > 0:
                ax.text(j, i, str(valor), ha="center", va="center",
                         color="white" if valor > matriz.max() / 2 else "black", fontsize=7)
    fig.colorbar(im, fraction=0.046)
    plt.tight_layout()
    plt.savefig(ruta, dpi=150)
    plt.close()


def graficar_curva_comparativa(curvas_val, ruta_salida):
    plt.figure(figsize=(8, 4.5))
    for nombre, historial in curvas_val.items():
        plt.plot(historial, color=COLORES_VARIANTES[nombre], label=nombre.upper())
    plt.yscale("log")
    plt.title("SGD vs Adam (Keras CNN): perdida de validacion por epoca (muestra reducida, full-batch)",
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
            print(f"\n=== Entrenando '{nombre}' (Keras CNN, full-batch) ===")
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
