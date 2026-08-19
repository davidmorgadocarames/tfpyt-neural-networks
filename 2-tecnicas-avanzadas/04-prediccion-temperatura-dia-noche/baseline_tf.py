"""
Baseline (Keras) del problema de prediccion de temperatura dia/noche -- el script "main" de
este problema. Importa las constantes de adam.py/lr_decay.py/weight_decay.py y monta UN
optimizador AdamW con LR-schedule y weight decay ya combinados -- ver
../03-tipos-clientes/baseline_tf.py para la explicacion completa del diseno.

Mismos datos sinteticos que RRNN/04-prediccion-temperatura-dia-noche (8 dias x 24 lecturas
horarias = 192 puntos, ciclo dia/noche + ruido gaussiano, SEED_DATOS=42), split CRONOLOGICO
60/20/20 (nunca aleatorio) y normalizacion min-max con min/max de SOLO el tramo de train.
Diferencia deliberada frente al original: RRNN aplana la ventana de VENTANA=3 horas a un vector
para una red densa; aqui se trata como una secuencia real, con una LSTM
(LSTM(32) -> Dense(16, relu) -> Dense(1)).

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

SEED_DATOS = 42
N_DIAS = 8
PUNTOS_POR_DIA = 24
TEMP_MEDIA_C = 18.0
AMPLITUD_C = 7.0
RUIDO_STD_C = 0.7
VENTANA = 3  # horas de historia que mira la red para predecir la siguiente

EPOCHS_MAX = 3000
PATIENCE = 200
MIN_DELTA = 1e-6  # perdida MSE sobre datos normalizados a [0,1]: escala mucho mas pequena que
# la entropia cruzada de los problemas de clasificacion, de ahi el min_delta mas pequeno


def generar_serie():
    """Identica al generador de RRNN/04-prediccion-temperatura-dia-noche -- ciclo dia/noche
    (coseno, minimo a medianoche, maximo a mediodia) + ruido gaussiano, SIEMPRE con
    SEED_DATOS."""
    np.random.seed(SEED_DATOS)
    n_puntos = N_DIAS * PUNTOS_POR_DIA
    dias = np.linspace(0, N_DIAS, n_puntos, endpoint=False)
    fase = 2 * np.pi * dias
    temperatura_c = TEMP_MEDIA_C - AMPLITUD_C * np.cos(fase) + np.random.normal(0, RUIDO_STD_C, n_puntos)
    return dias, temperatura_c


def cargar_datos(quiet=False):
    """Split cronologico 60/20/20 + ventaneo + normalizacion min-max con min/max de SOLO
    train, calculado ANTES de normalizar."""
    dias, temperatura_c = generar_serie()

    n = len(temperatura_c) - VENTANA
    split_train = int(n * 0.6)
    split_val = int(n * 0.8)
    raw_fin_train = split_train + VENTANA

    t_min, t_max = np.min(temperatura_c[:raw_fin_train]), np.max(temperatura_c[:raw_fin_train])
    serie_norm = (temperatura_c - t_min) / (t_max - t_min)

    X_lista, Y_lista = [], []
    for i in range(len(serie_norm) - VENTANA):
        X_lista.append(serie_norm[i : i + VENTANA])
        Y_lista.append(serie_norm[i + VENTANA])
    X = np.array(X_lista, dtype="float32").reshape(-1, VENTANA, 1)
    Y = np.array(Y_lista, dtype="float32").reshape(-1, 1)

    X_train, X_val, X_test = X[:split_train], X[split_train:split_val], X[split_val:]
    Y_train, Y_val, Y_test = Y[:split_train], Y[split_train:split_val], Y[split_val:]

    if not quiet:
        print(f"Temperatura: {len(X_train)} train / {len(X_val)} val / {len(X_test)} test (ventana={VENTANA}h)")
    return X_train, Y_train, X_val, Y_val, X_test, Y_test, t_min, t_max, dias, split_val


def build_model() -> tf.keras.Model:
    return tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(VENTANA, 1)),
            tf.keras.layers.LSTM(32),
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dense(1),
        ]
    )


def crear_optimizador() -> tf.keras.optimizers.Optimizer:
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=LEARNING_RATE_ADAM, decay_steps=DECAY_EVERY, decay_rate=DECAY_RATE, staircase=True)
    return tf.keras.optimizers.AdamW(learning_rate=lr_schedule, weight_decay=WEIGHT_DECAY)


def graficar_curva(historial_loss_val, ruta_salida):
    plt.figure(figsize=(7, 4.5))
    plt.plot(historial_loss_val, color="#4C72B0")
    plt.yscale("log")
    plt.title("Baseline LSTM (Adam+LRdecay+WeightDecay): MSE de validacion por epoca", fontweight="bold")
    plt.xlabel("Epoca")
    plt.ylabel("MSE de validacion, normalizado (escala log)")
    plt.grid(True, which="both", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(ruta_salida, dpi=150)
    plt.close()


def graficar_prediccion(y_test_c, y_pred_c, horas_test, ruta_salida):
    plt.figure(figsize=(8, 4.5))
    plt.plot(horas_test, y_test_c, label="Temperatura real", color="#4C72B0")
    plt.plot(horas_test, y_pred_c, label="Temperatura predicha", color="#C44E52", alpha=0.8)
    plt.xlabel("Horas dentro del tramo de test")
    plt.ylabel("Temperatura (°C)")
    plt.title("Baseline LSTM: temperatura real vs predicha (test)", fontweight="bold")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(ruta_salida, dpi=150)
    plt.close()


def main(seed_split=42, seed_modelo=42, quiet=False, guardar_graficas=True) -> dict:
    x_train, y_train, x_val, y_val, x_test, y_test, t_min, t_max, dias, split_val = cargar_datos(quiet=quiet)

    tf.keras.utils.set_random_seed(seed_modelo)
    tf.config.experimental.enable_op_determinism()

    model = build_model()
    model.compile(optimizer=crear_optimizador(), loss="mse", metrics=["mae"])

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

    y_pred_norm = model.predict(x_test, verbose=0)
    y_test_c = y_test.flatten() * (t_max - t_min) + t_min
    y_pred_c = y_pred_norm.flatten() * (t_max - t_min) + t_min
    mae_test_celsius = float(np.mean(np.abs(y_pred_c - y_test_c)))
    rmse_test_celsius = float(np.sqrt(np.mean((y_pred_c - y_test_c) ** 2)))
    loss_test = float(model.evaluate(x_test, y_test, verbose=0)[0])

    if not quiet:
        print(f"MAE test: {mae_test_celsius:.3f} C -- epocas: {epocas_entrenadas} "
              f"(mejor val en {epoca_mejor + 1}) -- {tiempo_computacion_segundos:.1f}s")

    horas_test = (dias[VENTANA + split_val : VENTANA + split_val + len(y_test)]
                  - dias[VENTANA + split_val]) * 24

    metrics = {
        "seed_modelo": seed_modelo,
        "ventana_horas": VENTANA,
        "learning_rate_adam": LEARNING_RATE_ADAM,
        "decay_rate": DECAY_RATE,
        "decay_every": DECAY_EVERY,
        "weight_decay": WEIGHT_DECAY,
        "epochs_max_configuradas": EPOCHS_MAX,
        "epochs_entrenadas": epocas_entrenadas,
        "epoca_mejor_val": epoca_mejor + 1,
        "loss_val_final": float(val_loss_hist[epoca_mejor]),
        "loss_test_final": loss_test,
        "mae_test_celsius": mae_test_celsius,
        "rmse_test_celsius": rmse_test_celsius,
        "tiempo_computacion_segundos": tiempo_computacion_segundos,
        "n_train": len(x_train), "n_val": len(x_val), "n_test": len(x_test),
        "historial_loss_val": val_loss_hist,
        "y_test_celsius": y_test_c.tolist(),
        "y_pred_celsius": y_pred_c.tolist(),
        "horas_test": horas_test.tolist(),
    }

    if not guardar_graficas:
        return metrics

    (RESULTS_DIR / "baseline_tf_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    graficar_curva(val_loss_hist, RESULTS_DIR / "baseline_tf_curva.png")
    graficar_prediccion(y_test_c, y_pred_c, horas_test, RESULTS_DIR / "baseline_tf_prediccion.png")

    if not quiet:
        print(f"Resultados guardados en {RESULTS_DIR}")

    return metrics


if __name__ == "__main__":
    main()
