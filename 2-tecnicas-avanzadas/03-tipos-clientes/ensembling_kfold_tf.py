"""
Ensembling + validacion cruzada k-fold combinados (Keras) -- en vez de descartar los K modelos
de un k-fold tras medir su accuracy de validacion, aqui se CONSERVAN y se promedian sus
predicciones. Cada uno de los K modelos usa la misma configuracion combinada de
baseline_tf.py (arquitectura + AdamW con LR-schedule + weight decay).

Diseno para evitar fuga de datos: split 80/20 en train_pool / test_final -- test_final queda
completamente al margen de cualquier fold, se usa una unica vez al final para evaluar tanto la
accuracy media de los K modelos individuales como la del ensemble. K-fold (K=5, estratificado)
sobre train_pool -- cada fold entrena un modelo usando el resto de train_pool como train y su
propio slice como validacion (early stopping), y reporta su accuracy en ese slice (informe
k-fold clasico).

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
from sklearn.model_selection import StratifiedKFold, train_test_split

from adam import LEARNING_RATE_ADAM
from lr_decay import DECAY_EVERY, DECAY_RATE
from weight_decay import WEIGHT_DECAY
from baseline_tf import EPOCHS_MAX, MIN_DELTA, PATIENCE, build_model, generar_datos, graficar_confusion

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

K_FOLDS = 5
SEED_SPLIT_FINAL = 42  # split train_pool/test_final
SEED_KFOLD = 42  # reparto de folds dentro de train_pool
SEED_MODELO = 42


def crear_optimizador():
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=LEARNING_RATE_ADAM, decay_steps=DECAY_EVERY, decay_rate=DECAY_RATE, staircase=True)
    return tf.keras.optimizers.AdamW(learning_rate=lr_schedule, weight_decay=WEIGHT_DECAY)


def main(quiet=False) -> dict:
    X, Y_num = generar_datos()

    idx_pool, idx_test_final = train_test_split(
        np.arange(len(X)), test_size=0.2, stratify=Y_num, random_state=SEED_SPLIT_FINAL)
    X_pool, Y_pool = X[idx_pool], Y_num[idx_pool]
    X_test_final_raw, Y_test_final = X[idx_test_final], Y_num[idx_test_final]

    skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=SEED_KFOLD)
    accuracies_fold = []
    probs_por_modelo = []
    tiempo_total = 0.0

    for fold, (idx_train, idx_val) in enumerate(skf.split(X_pool, Y_pool)):
        X_train_raw, X_val_raw = X_pool[idx_train], X_pool[idx_val]
        Y_train, Y_val = Y_pool[idx_train], Y_pool[idx_val]

        # Normalizacion con min/max de ESTE fold de train (nunca de val ni de test_final).
        X_min, X_max = X_train_raw.min(axis=0), X_train_raw.max(axis=0)
        X_train = ((X_train_raw - X_min) / (X_max - X_min)).astype("float32")
        X_val = ((X_val_raw - X_min) / (X_max - X_min)).astype("float32")
        X_test_final = ((X_test_final_raw - X_min) / (X_max - X_min)).astype("float32")

        tf.keras.utils.set_random_seed(SEED_MODELO + fold)
        tf.config.experimental.enable_op_determinism()

        model = build_model()
        model.compile(optimizer=crear_optimizador(), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
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

        _, acc_val = model.evaluate(X_val, Y_val, verbose=0)
        accuracies_fold.append(float(acc_val))
        probs_por_modelo.append(model.predict(X_test_final, verbose=0))

        if not quiet:
            print(f"Fold {fold + 1}/{K_FOLDS}: accuracy val = {acc_val:.4f} ({len(idx_val)} muestras)")

    media_fold = float(np.mean(accuracies_fold))
    desviacion_fold = float(np.std(accuracies_fold))

    accuracies_individuales_test = [
        float(np.mean(np.argmax(p, axis=1) == Y_test_final)) for p in probs_por_modelo
    ]
    probs_ensemble = np.mean(probs_por_modelo, axis=0)
    pred_ensemble = np.argmax(probs_ensemble, axis=1)
    accuracy_ensemble = float(np.mean(pred_ensemble == Y_test_final))

    matriz_confusion = np.zeros((3, 3), dtype=int)
    for real, pred in zip(Y_test_final, pred_ensemble):
        matriz_confusion[real, pred] += 1

    metrics = {
        "k_folds": K_FOLDS,
        "seed_split_final": SEED_SPLIT_FINAL,
        "seed_kfold": SEED_KFOLD,
        "seed_modelo": SEED_MODELO,
        "n_pool": len(X_pool),
        "n_test_final": len(X_test_final_raw),
        "accuracies_por_fold_val": accuracies_fold,
        "media_fold_val": media_fold,
        "desviacion_fold_val": desviacion_fold,
        "accuracies_individuales_test_final": accuracies_individuales_test,
        "media_individual_test_final": float(np.mean(accuracies_individuales_test)),
        "accuracy_ensemble_test_final": accuracy_ensemble,
        "tiempo_computacion_segundos": tiempo_total,
        "matriz_confusion_ensemble": matriz_confusion.tolist(),
    }
    (RESULTS_DIR / "ensembling_kfold_tf_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    plt.figure(figsize=(7, 4.5))
    xs = list(range(1, K_FOLDS + 1))
    plt.bar(xs, accuracies_individuales_test, color="#4C72B0", label="Modelos individuales (test_final)")
    plt.axhline(metrics["media_individual_test_final"], color="#4C72B0", linestyle=":",
                label=f"Media individual ({metrics['media_individual_test_final']:.4f})")
    plt.axhline(accuracy_ensemble, color="#C44E52", linestyle="--", linewidth=2,
                label=f"Ensemble ({accuracy_ensemble:.4f})")
    plt.xlabel("Fold (modelo)")
    plt.ylabel("Accuracy en test_final")
    plt.title(f"Ensembling + k-fold (k={K_FOLDS}) -- tipos de clientes (Keras)", fontweight="bold")
    plt.legend(fontsize=8)
    plt.grid(True, axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "ensembling_kfold_tf_accuracy.png", dpi=150)
    plt.close()

    graficar_confusion(matriz_confusion, accuracy_ensemble, "Matriz de confusion -- ensemble sobre test_final",
                        RESULTS_DIR / "ensembling_kfold_tf_confusion.png")

    if not quiet:
        print(f"\nAccuracy media por fold (val): {media_fold:.4f} +/- {desviacion_fold:.4f}")
        print(f"Accuracy media individual (test_final): {metrics['media_individual_test_final']:.4f}")
        print(f"Accuracy ensemble (test_final): {accuracy_ensemble:.4f}")
        print(f"Resultados guardados en {RESULTS_DIR}")

    return metrics


if __name__ == "__main__":
    main()
