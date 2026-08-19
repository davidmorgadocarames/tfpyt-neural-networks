"""
Ensembling + validacion cruzada combinados (Keras) sobre temperatura dia/noche -- a diferencia
de ../03-tipos-clientes/ y ../06-zonas-espirales/, aqui NO se puede usar k-fold estratificado
normal: barajar una serie temporal para repartirla en folds entrenaria con datos del futuro
para predecir el pasado. Se usa `sklearn.model_selection.TimeSeriesSplit` -- cada fold entrena
con una ventana creciente de pasado y valida con el tramo inmediatamente siguiente, sin
barajar. Cada modelo usa la config combinada de baseline_tf.py (AdamW+LRdecay+WeightDecay+LSTM).

Diseno: el ultimo 20% de la serie (cronologicamente) se reserva como test_final, igual que en
baseline_tf.py -- jamas lo ve ningun fold. TimeSeriesSplit(K_FOLDS) reparte el 80% restante
(train_pool) en K pares (train, val) de ventana creciente. Los K modelos se conservan y se
promedian sus predicciones sobre test_final.

Uso: python ensembling_kfold_tf.py
"""

import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.model_selection import TimeSeriesSplit

from adam import LEARNING_RATE_ADAM
from lr_decay import DECAY_EVERY, DECAY_RATE
from weight_decay import WEIGHT_DECAY
from baseline_tf import EPOCHS_MAX, MIN_DELTA, PATIENCE, VENTANA, build_model, generar_serie

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

K_FOLDS = 5
SEED_MODELO = 42


def crear_optimizador():
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=LEARNING_RATE_ADAM, decay_steps=DECAY_EVERY, decay_rate=DECAY_RATE, staircase=True)
    return tf.keras.optimizers.AdamW(learning_rate=lr_schedule, weight_decay=WEIGHT_DECAY)


def main(quiet=False) -> dict:
    dias, temperatura_c = generar_serie()

    n = len(temperatura_c) - VENTANA
    split_pool_final = int(n * 0.8)
    raw_fin_pool = split_pool_final + VENTANA

    t_min, t_max = np.min(temperatura_c[:raw_fin_pool]), np.max(temperatura_c[:raw_fin_pool])
    serie_norm = (temperatura_c - t_min) / (t_max - t_min)

    X_lista, Y_lista = [], []
    for i in range(len(serie_norm) - VENTANA):
        X_lista.append(serie_norm[i : i + VENTANA])
        Y_lista.append(serie_norm[i + VENTANA])
    X = np.array(X_lista, dtype="float32").reshape(-1, VENTANA, 1)
    Y = np.array(Y_lista, dtype="float32").reshape(-1, 1)

    X_pool, Y_pool = X[:split_pool_final], Y[:split_pool_final]
    X_test_final, Y_test_final = X[split_pool_final:], Y[split_pool_final:]
    Y_test_final_c = Y_test_final.flatten() * (t_max - t_min) + t_min

    tscv = TimeSeriesSplit(n_splits=K_FOLDS)
    mae_fold = []
    preds_por_modelo = []
    tiempo_total = 0.0

    for fold, (idx_train, idx_val) in enumerate(tscv.split(X_pool)):
        X_train, Y_train = X_pool[idx_train], Y_pool[idx_train]
        X_val, Y_val = X_pool[idx_val], Y_pool[idx_val]

        tf.keras.utils.set_random_seed(SEED_MODELO + fold)
        tf.config.experimental.enable_op_determinism()

        model = build_model()
        model.compile(optimizer=crear_optimizador(), loss="mse", metrics=["mae"])
        t0 = time.time()
        model.fit(
            X_train, Y_train,
            validation_data=(X_val, Y_val),
            epochs=EPOCHS_MAX,
            batch_size=len(X_train),
            callbacks=[tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=PATIENCE, min_delta=MIN_DELTA, restore_best_weights=True)],
            verbose=0,
        )
        tiempo_total += time.time() - t0

        pred_val_norm = model.predict(X_val, verbose=0)
        y_val_c = Y_val.flatten() * (t_max - t_min) + t_min
        pred_val_c = pred_val_norm.flatten() * (t_max - t_min) + t_min
        mae_val = float(np.mean(np.abs(pred_val_c - y_val_c)))
        mae_fold.append(mae_val)

        preds_por_modelo.append(model.predict(X_test_final, verbose=0).flatten())

        if not quiet:
            print(f"Fold {fold + 1}/{K_FOLDS}: MAE val = {mae_val:.3f} C "
                  f"(train={len(idx_train)}, val={len(idx_val)})")

    media_fold = float(np.mean(mae_fold))
    desviacion_fold = float(np.std(mae_fold))

    preds_test_c = [(p * (t_max - t_min) + t_min) for p in preds_por_modelo]
    mae_individuales_test = [float(np.mean(np.abs(p - Y_test_final_c))) for p in preds_test_c]
    pred_ensemble_c = np.mean(preds_test_c, axis=0)
    mae_ensemble_test = float(np.mean(np.abs(pred_ensemble_c - Y_test_final_c)))

    metrics = {
        "k_folds": K_FOLDS,
        "seed_modelo": SEED_MODELO,
        "n_pool": len(X_pool),
        "n_test_final": len(X_test_final),
        "mae_por_fold_val_celsius": mae_fold,
        "media_fold_val_celsius": media_fold,
        "desviacion_fold_val_celsius": desviacion_fold,
        "mae_individuales_test_final_celsius": mae_individuales_test,
        "media_individual_test_final_celsius": float(np.mean(mae_individuales_test)),
        "mae_ensemble_test_final_celsius": mae_ensemble_test,
        "tiempo_computacion_segundos": tiempo_total,
        "y_test_final_celsius": Y_test_final_c.tolist(),
        "pred_ensemble_celsius": pred_ensemble_c.tolist(),
    }
    (RESULTS_DIR / "ensembling_kfold_tf_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    plt.figure(figsize=(7, 4.5))
    xs = list(range(1, K_FOLDS + 1))
    plt.bar(xs, mae_individuales_test, color="#4C72B0", label="Modelos individuales (test_final)")
    plt.axhline(metrics["media_individual_test_final_celsius"], color="#4C72B0", linestyle=":",
                label=f"Media individual ({metrics['media_individual_test_final_celsius']:.3f}C)")
    plt.axhline(mae_ensemble_test, color="#C44E52", linestyle="--", linewidth=2,
                label=f"Ensemble ({mae_ensemble_test:.3f}C)")
    plt.xlabel("Fold (modelo)")
    plt.ylabel("MAE en test_final (C)")
    plt.title(f"Ensembling + TimeSeriesSplit (k={K_FOLDS}) -- temperatura (Keras)", fontweight="bold")
    plt.legend(fontsize=8)
    plt.grid(True, axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "ensembling_kfold_tf_mae.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 4.5))
    plt.plot(Y_test_final_c, label="Temperatura real", color="#4C72B0")
    plt.plot(pred_ensemble_c, label="Temperatura predicha (ensemble)", color="#C44E52", alpha=0.8)
    plt.xlabel("Muestras de test_final (orden cronologico)")
    plt.ylabel("Temperatura (°C)")
    plt.title("Ensemble: temperatura real vs predicha (test_final)", fontweight="bold")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "ensembling_kfold_tf_prediccion.png", dpi=150)
    plt.close()

    if not quiet:
        print(f"\nMAE medio por fold (val): {media_fold:.3f} +/- {desviacion_fold:.3f} C")
        print(f"MAE medio individual (test_final): {metrics['media_individual_test_final_celsius']:.3f} C")
        print(f"MAE ensemble (test_final): {mae_ensemble_test:.3f} C")
        print(f"Resultados guardados en {RESULTS_DIR}")

    return metrics


if __name__ == "__main__":
    main()
