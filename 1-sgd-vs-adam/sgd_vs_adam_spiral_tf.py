"""
SGD vs Adam (Keras) sobre el problema de las zonas de espirales -- equivalente TF de
RRNN/06-zonas-espirales/spiral_classifier.py: 3 brazos de una espiral (frontera de decision
curva, no separable linealmente), red densa mas profunda (2 -> 64 LeakyReLU -> 64 LeakyReLU ->
3 softmax), full-batch (270 muestras de train, 1 actualizacion de pesos por epoca).

Mismos datos que el original: 450 puntos (150/brazo) generados SIEMPRE con SEED_DATOS=0 via la
API legacy de np.random (mismo generador r/theta con ruido gaussiano de CS231n), split 60/20/20
estratificado por brazo con seed_split. A diferencia de tipos-clientes, sin normalizar -- los
datos ya viven en [-1, 1] por construccion (r in [0, 1]). Guarda resultados en
results_sgd_vs_adam/spiral_tf/.

Uso: python sgd_vs_adam_spiral_tf.py
"""

import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

RESULTS_DIR = Path(__file__).parent / "results_sgd_vs_adam" / "spiral_tf"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SEED_DATOS = 0  # gobierna solo la generacion de datos -- fijo siempre, igual que en RRNN
N_POR_BRAZO = 150
K_CLASES = 3
EPOCHS_MAX = 5000
PATIENCE = 200  # igual que PACIENCIA_EARLY_STOP en spiral_classifier.py
MIN_DELTA = 1e-4  # ver el comentario en sgd_vs_adam_customer_tf.py -- sin Dropout, full-batch,
# determinista: hace falta min_delta para que el plateau dispare el EarlyStopping.

VARIANTES = [
    ("sgd", lambda: tf.keras.optimizers.SGD(learning_rate=1.0, momentum=0.0)),
    ("adam", lambda: tf.keras.optimizers.Adam(learning_rate=0.01)),
]
COLORES_VARIANTES = {"sgd": "#4C72B0", "adam": "#55A868"}


def generar_datos():
    """Identico a generar_datos() en spiral_classifier.py -- adaptado del "minimal neural
    network case study" de CS231n (Andrej Karpathy), SIEMPRE con SEED_DATOS."""
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
    """Identico a split_estratificado() en spiral_classifier.py."""
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


def build_model(optimizer) -> tf.keras.Model:
    """Misma arquitectura que spiral_classifier.py: 2 -> 64 LeakyReLU -> 64 LeakyReLU -> 3
    softmax, sin Dropout (el original tampoco lo usa)."""
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(2,)),
            tf.keras.layers.Dense(64, activation=tf.keras.layers.LeakyReLU(negative_slope=0.01)),
            tf.keras.layers.Dense(64, activation=tf.keras.layers.LeakyReLU(negative_slope=0.01)),
            tf.keras.layers.Dense(3, activation="softmax"),
        ]
    )
    model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


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


def graficar_curva_comparativa(curvas_val, ruta_salida):
    plt.figure(figsize=(8, 4.5))
    for nombre, historial in curvas_val.items():
        plt.plot(historial, color=COLORES_VARIANTES[nombre], label=nombre.upper())
    plt.yscale("log")
    plt.title("SGD vs Adam (Keras): perdida de validacion por epoca (zonas de espirales)",
               fontweight="bold")
    plt.xlabel("Epoca")
    plt.ylabel("Perdida de validacion (escala log)")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(ruta_salida, dpi=150)
    plt.close()


def main(seed_split=0, seed_modelo=0, quiet=False, guardar_graficas=True) -> dict:
    x_train, y_train, x_val, y_val, x_test, y_test = cargar_datos(seed_split, quiet=quiet)

    resultados = {}
    curvas_val = {}
    for nombre, crear_optimizador in VARIANTES:
        if not quiet:
            print(f"\n=== Entrenando '{nombre}' (Keras, zonas-espirales, full-batch) ===")
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
                monitor="val_loss", patience=PATIENCE, min_delta=MIN_DELTA, restore_best_weights=True)],
            verbose=2 if not quiet else 0,
        )
        tiempo = time.time() - t0

        val_loss_hist = history.history["val_loss"]
        epoca_mejor = int(np.argmin(val_loss_hist))
        epocas_entrenadas = len(val_loss_hist)

        test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
        y_pred = np.argmax(model.predict(x_test, verbose=0), axis=1)
        matriz_confusion = np.zeros((3, 3), dtype=int)
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
