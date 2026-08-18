"""
LSTM (Keras) para mantenimiento predictivo: estima la vida util restante (RUL, Remaining
Useful Life) de una maquina a partir de series temporales de sensores. Reaprovecha la idea del
antiguo 03_lstm_predictive_maintenance.py (eliminado de 01-tensorflow-vision-series/ al
reorganizar el proyecto en 1-sgd-vs-adam/ y 2-metodologia-ml/) en vez de partir de cero.

El dataset es sintetico pero esta generado para imitar la estructura del NASA C-MAPSS Turbofan
Engine Degradation dataset (multiples "motores", varios sensores con degradacion no lineal mas
ruido, hasta el fallo) para que el pipeline sea reproducible sin depender de una descarga
externa. El patron de degradacion (forma cuadratica, amplitud por sensor) es compartido entre
unidades -- como una flota real de maquinas del mismo modelo -- con solo una pequena variacion
de calibracion por unidad; si cada unidad tuviese una amplitud de degradacion completamente
aleatoria e independiente, el RUL dejaria de ser inferible a partir de los sensores.

A diferencia del script original, aqui el JSON de resultados guarda tambien los datos crudos
que alimentan las dos graficas (historial de MAE por epoca completo, y los pares RUL
real/predicho de cada ventana de test) -- regla de reproducibilidad de este proyecto: sin eso,
una grafica mal coloreada o que se quiera rehacer con otro estilo exigiria reentrenar desde
cero en vez de releer el JSON.

Uso: python lstm_mantenimiento_predictivo.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

SEED = 42
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

N_UNITS = 80  # numero de "motores"/maquinas simuladas
N_SENSORS = 4  # sensores por maquina
WINDOW = 20  # tamano de ventana temporal para el LSTM
MAX_RUL_CLIP = 130  # clipping estandar en literatura de C-MAPSS (RUL se aplana al inicio)


def simulate_run_to_failure(rng: np.random.Generator) -> list[np.ndarray]:
    """Genera series temporales sensor x tiempo hasta el fallo, con degradacion no lineal
    compartida entre unidades (ver docstring del modulo)."""
    common_base = rng.uniform(0.3, 0.7, size=N_SENSORS)
    common_drift = rng.uniform(0.8, 1.2, size=N_SENSORS)

    units = []
    for _ in range(N_UNITS):
        life = rng.integers(120, 260)
        t = np.arange(life)
        degradation = (t / life) ** 2  # degradacion acelerada cerca del fallo
        base = common_base + rng.normal(0, 0.02, size=N_SENSORS)
        drift = common_drift * rng.uniform(0.9, 1.1, size=N_SENSORS)
        sensors = np.zeros((life, N_SENSORS))
        for s in range(N_SENSORS):
            noise = rng.normal(0, 0.02, size=life)
            sensors[:, s] = base[s] + drift[s] * degradation + noise
        units.append(sensors)
    return units


def make_windows(units: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for sensors in units:
        life = sensors.shape[0]
        rul = np.clip(life - 1 - np.arange(life), 0, MAX_RUL_CLIP)
        for i in range(life - WINDOW):
            X.append(sensors[i : i + WINDOW])
            y.append(rul[i + WINDOW - 1])
    return np.array(X, dtype="float32"), np.array(y, dtype="float32")


def build_model() -> tf.keras.Model:
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(WINDOW, N_SENSORS)),
            tf.keras.layers.LSTM(64, return_sequences=True),
            tf.keras.layers.LSTM(32),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(1),
        ],
        name="lstm_rul_predictive_maintenance",
    )
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


def main(quiet=False) -> dict:
    rng = np.random.default_rng(SEED)
    tf.keras.utils.set_random_seed(SEED)
    tf.config.experimental.enable_op_determinism()

    units = simulate_run_to_failure(rng)
    split = int(0.8 * N_UNITS)
    X_train, y_train = make_windows(units[:split])
    X_test, y_test = make_windows(units[split:])

    scaler = StandardScaler().fit(X_train.reshape(-1, N_SENSORS))
    X_train = scaler.transform(X_train.reshape(-1, N_SENSORS)).reshape(X_train.shape).astype("float32")
    X_test = scaler.transform(X_test.reshape(-1, N_SENSORS)).reshape(X_test.shape).astype("float32")

    model = build_model()
    if not quiet:
        model.summary()

    history = model.fit(
        X_train,
        y_train,
        validation_split=0.1,
        epochs=30,
        batch_size=64,
        callbacks=[tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)],
        verbose=2 if not quiet else 0,
    )

    y_pred = model.predict(X_test, verbose=0).flatten()
    mae = mean_absolute_error(y_test, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))

    metrics = {
        "seed": SEED,
        "mae_ciclos": float(mae),
        "rmse_ciclos": rmse,
        "train_mae_ciclos": float(history.history["mae"][-1]),
        "epochs_run": len(history.history["loss"]),
        "n_test_windows": int(len(y_test)),
        # datos crudos que alimentan las graficas -- regla de reproducibilidad del proyecto
        "historial": {k: [float(v) for v in vs] for k, vs in history.history.items()},
        "y_test": y_test.tolist(),
        "y_pred": y_pred.tolist(),
    }
    (RESULTS_DIR / "lstm_rul_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    model.save(RESULTS_DIR / "lstm_rul_predictive_maintenance.keras")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(history.history["mae"], label="train")
    axes[0].plot(history.history["val_mae"], label="val")
    axes[0].set_title("MAE por epoca (ciclos de RUL)")
    axes[0].set_xlabel("epoca")
    axes[0].legend()

    order = np.argsort(y_test)
    axes[1].plot(y_test[order], label="RUL real")
    axes[1].plot(y_pred[order], label="RUL predicho", alpha=0.7)
    axes[1].set_title(f"RUL real vs predicho (MAE={mae:.1f} ciclos)")
    axes[1].set_xlabel("ventanas de test (ordenadas por RUL real)")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "lstm_rul_training.png", dpi=150)
    plt.close(fig)

    if not quiet:
        print(f"\nMAE test: {mae:.2f} ciclos | RMSE test: {rmse:.2f} ciclos")
        print(f"Resultados guardados en {RESULTS_DIR}")
    return metrics


if __name__ == "__main__":
    main()
