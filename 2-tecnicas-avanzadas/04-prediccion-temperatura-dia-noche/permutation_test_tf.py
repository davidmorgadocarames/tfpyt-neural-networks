"""
Test de permutacion (Keras) -- replica RRNN/04-prediccion-temperatura-dia-noche/permutation_test.py
(leido integramente) con la config combinada de baseline_tf.py (AdamW+LRdecay+WeightDecay+LSTM).

Pregunta: ¿la red aprendio el ciclo dia/noche real, o llegaria a un error parecido con
cualquier ruido que tuviera la misma media y varianza? Distinta de "Robustez frente a la
semilla" -- esa varia la inicializacion de pesos sobre los MISMOS datos; esta mantiene la
inicializacion FIJA (misma seed_modelo en todas las repeticiones) y varia el ORDEN de los
datos, para medir si el patron que explota la red es genuino.

Metodologia: se toma la serie de 192 temperaturas reales y se baraja por completo (mismos
valores, misma media/varianza, sin ciclo dia/noche) N veces con ordenes distintos. Cada
barajado se entrena con la misma receta exacta (misma arquitectura, mismo split 60/20/20
cronologico por posicion, misma normalizacion min-max de train, mismo AdamW+LRdecay+
WeightDecay) y SIEMPRE con la misma seed_modelo. Si el MAE real queda muy por debajo de la
distribucion de MAEs barajados, la red esta explotando el ciclo dia/noche genuino.

Nota de coste computacional: RRNN usa N=1000 por defecto sobre una red densa NumPy que entrena
en milisegundos; aqui cada entrenamiento es una LSTM en TF/Keras (~1-2 min con esta
configuracion), por lo que N=1000 tardaria horas. El valor por defecto de este script es mucho
mas bajo (ver N_PERMUTACIONES_DEFAULT) -- documentado como tradeoff deliberado, con --n para
ampliarlo si se dispone de mas tiempo de computo.

Uso:
    python permutation_test_tf.py              # N por defecto
    python permutation_test_tf.py --n 20        # para probar rapido cuanto tarda
"""

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from baseline_tf import (
    AMPLITUD_C,
    EPOCHS_MAX,
    MIN_DELTA,
    N_DIAS,
    PATIENCE,
    PUNTOS_POR_DIA,
    RUIDO_STD_C,
    SEED_DATOS,
    TEMP_MEDIA_C,
    VENTANA,
    build_model,
    crear_optimizador,
)

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

SEED_MODELO_FIJO = 42  # misma seed_modelo en TODAS las repeticiones -- solo cambia el orden
SEED_PERMUTACION = 2024
N_PERMUTACIONES_DEFAULT = 100  # ver nota de coste computacional en el docstring del modulo


def generar_serie_real():
    """Misma generacion que baseline_tf.py, con SEED_DATOS."""
    np.random.seed(SEED_DATOS)
    n_puntos = N_DIAS * PUNTOS_POR_DIA
    dias = np.linspace(0, N_DIAS, n_puntos, endpoint=False)
    fase = 2 * np.pi * dias
    return TEMP_MEDIA_C - AMPLITUD_C * np.cos(fase) + np.random.normal(0, RUIDO_STD_C, n_puntos)


def entrenar_evaluar(serie_temperatura, seed_modelo):
    """Misma receta que baseline_tf.py (ventana deslizante, split 60/20/20 cronologico,
    normalizacion min-max de train, AdamW+LRdecay+WeightDecay+LSTM) pero recibiendo la serie
    ya barajada (o la real) como parametro."""
    n_puntos = len(serie_temperatura)
    n = n_puntos - VENTANA
    split_train = int(n * 0.6)
    split_val = int(n * 0.8)
    raw_fin_train = split_train + VENTANA

    t_min = np.min(serie_temperatura[:raw_fin_train])
    t_max = np.max(serie_temperatura[:raw_fin_train])
    serie_norm = (serie_temperatura - t_min) / (t_max - t_min)

    X_lista, Y_lista = [], []
    for i in range(len(serie_norm) - VENTANA):
        X_lista.append(serie_norm[i : i + VENTANA])
        Y_lista.append(serie_norm[i + VENTANA])
    X = np.array(X_lista, dtype="float32").reshape(-1, VENTANA, 1)
    Y = np.array(Y_lista, dtype="float32").reshape(-1, 1)

    X_train, X_val, X_test = X[:split_train], X[split_train:split_val], X[split_val:]
    Y_train, Y_val, Y_test = Y[:split_train], Y[split_train:split_val], Y[split_val:]

    tf.keras.utils.set_random_seed(seed_modelo)
    tf.config.experimental.enable_op_determinism()

    model = build_model()
    model.compile(optimizer=crear_optimizador(), loss="mse", metrics=["mae"])
    model.fit(
        X_train, Y_train,
        validation_data=(X_val, Y_val),
        epochs=EPOCHS_MAX,
        batch_size=len(X_train),
        callbacks=[tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=PATIENCE, min_delta=MIN_DELTA, restore_best_weights=True)],
        verbose=0,
    )

    y_pred_norm = model.predict(X_test, verbose=0)
    y_test_c = Y_test.flatten() * (t_max - t_min) + t_min
    y_pred_c = y_pred_norm.flatten() * (t_max - t_min) + t_min
    return float(np.mean(np.abs(y_pred_c - y_test_c)))


def graficar_histograma(mae_permutados, mae_real, p_valor, ruta_salida):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(mae_permutados, bins=min(40, len(mae_permutados)), color="#4C72B0", edgecolor="black",
            alpha=0.85, label=f"MAE con datos barajados (N={len(mae_permutados)})")
    ax.axvline(mae_real, color="#C44E52", linestyle="--", linewidth=2,
               label=f"MAE con datos reales = {mae_real:.2f} °C")
    cota_p = 1 / (len(mae_permutados) + 1)
    texto_p = f"p < {cota_p:.3g}" if p_valor <= cota_p else f"p = {p_valor:.3g}"
    ax.text(mae_real, ax.get_ylim()[1] * 0.92, f"  {texto_p}", color="#C44E52", fontsize=9, va="top")
    ax.set_xlabel("MAE en test (°C)")
    ax.set_ylabel("Nº de barajados")
    ax.set_title("Test de permutacion (Keras): ¿es real el ciclo dia/noche aprendido?",
                  fontweight="bold", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    plt.tight_layout()
    plt.savefig(ruta_salida, dpi=150)
    plt.close()


def main(n_permutaciones=N_PERMUTACIONES_DEFAULT):
    serie_real = generar_serie_real()

    print(f"Entrenando sobre datos reales (seed_modelo={SEED_MODELO_FIJO})...")
    t0 = time.time()
    mae_real = entrenar_evaluar(serie_real, SEED_MODELO_FIJO)
    print(f"MAE en test, datos reales: {mae_real:.4f} °C ({time.time() - t0:.0f}s)")

    rng_permutacion = np.random.default_rng(SEED_PERMUTACION)
    mae_permutados = []
    t0 = time.time()
    for i in range(n_permutaciones):
        serie_barajada = rng_permutacion.permutation(serie_real)
        mae = entrenar_evaluar(serie_barajada, SEED_MODELO_FIJO)
        mae_permutados.append(mae)
        print(f"  {i + 1}/{n_permutaciones} barajados -- {time.time() - t0:.0f}s acumulados", flush=True)

    mae_permutados = np.array(mae_permutados)
    n_iguala_o_mejora = int(np.sum(mae_permutados <= mae_real))
    p_valor = (n_iguala_o_mejora + 1) / (n_permutaciones + 1)

    resumen = {
        "seed_modelo_fijo": SEED_MODELO_FIJO,
        "seed_permutacion": SEED_PERMUTACION,
        "n_permutaciones": n_permutaciones,
        "mae_test_real_celsius": mae_real,
        "mae_test_permutado": {
            "media": float(np.mean(mae_permutados)),
            "desviacion_tipica": float(np.std(mae_permutados, ddof=1)) if n_permutaciones > 1 else 0.0,
            "min": float(np.min(mae_permutados)),
            "max": float(np.max(mae_permutados)),
            "valores": mae_permutados.tolist(),
        },
        "n_permutaciones_que_igualan_o_superan_lo_real": n_iguala_o_mejora,
        "p_valor_empirico": p_valor,
    }

    (RESULTS_DIR / "permutation_test_tf_metrics.json").write_text(
        json.dumps(resumen, indent=2, ensure_ascii=False), encoding="utf-8")
    graficar_histograma(mae_permutados, mae_real, p_valor, RESULTS_DIR / "permutation_test_tf.png")

    print(f"\nMAE real: {mae_real:.4f} °C")
    print(f"MAE barajado: media={resumen['mae_test_permutado']['media']:.4f}, "
          f"min={resumen['mae_test_permutado']['min']:.4f}, max={resumen['mae_test_permutado']['max']:.4f}")
    print(f"p-valor empirico: {p_valor:.4g} ({n_iguala_o_mejora}/{n_permutaciones} barajados "
          f"igualan o superan el resultado real)")
    print(f"Resultados guardados en {RESULTS_DIR}")

    return resumen


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=N_PERMUTACIONES_DEFAULT)
    args = parser.parse_args()
    main(n_permutaciones=args.n)
