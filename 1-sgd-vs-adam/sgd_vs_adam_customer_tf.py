"""
SGD vs Adam (Keras) sobre el problema de tipos de clientes -- equivalente TF de
RRNN/03-tipos-clientes/customer_classifier.py: 3 categorias de clientes (Navegadores,
Ocasionales, VIPs) a partir de 2 variables (minutos navegando, productos en carrito), red
densa pequena (2 -> 5 LeakyReLU -> 3 softmax), full-batch (72 muestras de train, 1
actualizacion de pesos por epoca).

Mismos datos que el original: 120 clientes (40/categoria) generados SIEMPRE con SEED_DATOS=42
via la API legacy de np.random, split 60/20/20 estratificado por categoria con seed_split, y
normalizacion min-max con min/max de train. Guarda resultados en
results_sgd_vs_adam/customer_tf/.

Uso: python sgd_vs_adam_customer_tf.py
"""

import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

RESULTS_DIR = Path(__file__).parent / "results_sgd_vs_adam" / "customer_tf"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SEED_DATOS = 42  # gobierna solo la generacion de datos -- fijo siempre, igual que en RRNN
N_TRAIN_CLASE, N_VAL_CLASE = 24, 8  # 24 train / 8 val / 8 test por categoria (60/20/20)
NOMBRES_CLASES = ["Navegadores", "Ocasionales", "VIPs"]
EPOCHS_MAX = 3000
PATIENCE = 200  # igual que PACIENCIA_EARLY_STOP en customer_classifier.py
MIN_DELTA = 1e-4  # sin Dropout el entrenamiento full-batch es determinista: la perdida de
# validacion baja de forma monotona en fracciones minusculas indefinidamente, asi que un
# EarlyStopping con min_delta=0 (el criterio de "mejora" por defecto de Keras) nunca detecta
# el plateau y siempre agota EPOCHS_MAX. min_delta exige una mejora de al menos esta magnitud
# para reiniciar la cuenta de paciencia -- necesario aqui, no en los 8 scripts de MNIST
# (Dropout(0.3) introduce ruido real en la perdida de validacion entre epocas).

VARIANTES = [
    ("sgd", lambda: tf.keras.optimizers.SGD(learning_rate=0.5, momentum=0.0)),
    ("adam", lambda: tf.keras.optimizers.Adam(learning_rate=0.02)),
]
COLORES_VARIANTES = {"sgd": "#4C72B0", "adam": "#55A868"}


def generar_datos():
    """Identico a customer_classifier.generar (dentro de main()): 120 clientes, 40 por
    categoria, SIEMPRE con SEED_DATOS."""
    np.random.seed(SEED_DATOS)
    X0 = np.random.uniform(2, 10, (40, 2)) + np.array([0, 0])
    X1 = np.random.uniform(15, 25, (40, 2)) + np.array([0, 2])
    X2 = np.random.uniform(30, 45, (40, 2)) + np.array([0, 6])
    X = np.vstack([X0, X1, X2])
    Y_num = np.concatenate([np.zeros(40), np.ones(40), np.full(40, 2)]).astype("int64")
    return X, Y_num


def cargar_datos(seed_split, quiet=False):
    """Split train/val/test estratificado (60/20/20) igual que customer_classifier.py, mas
    normalizacion min-max con min/max de train."""
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


def build_model(optimizer) -> tf.keras.Model:
    """Misma arquitectura que customer_classifier.py: 2 -> 5 LeakyReLU -> 3 softmax, sin
    Dropout (el original tampoco lo usa -- red demasiado pequena para necesitar regularizar)."""
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(2,)),
            tf.keras.layers.Dense(5, activation=tf.keras.layers.LeakyReLU(negative_slope=0.01)),
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


def graficar_curva_comparativa(curvas_val, ruta_salida):
    plt.figure(figsize=(8, 4.5))
    for nombre, historial in curvas_val.items():
        plt.plot(historial, color=COLORES_VARIANTES[nombre], label=nombre.upper())
    plt.yscale("log")
    plt.title("SGD vs Adam (Keras): perdida de validacion por epoca (tipos de clientes)",
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
            print(f"\n=== Entrenando '{nombre}' (Keras, tipos-clientes, full-batch) ===")
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
