"""
Ensembling + validacion cruzada k-fold combinados (Keras) sobre la CNN de Fashion-MNIST -- mismo
diseno que ../07-reconocimiento-digitos/ensembling_kfold_tf.py (split 80/20 train_pool/test_final
held-out, K=5 folds estratificados, los K modelos se conservan y se promedian sus predicciones).
Cada modelo usa la config combinada de baseline_tf.py (AdamW+LRdecay+WeightDecay+BatchNorm).

Usa una muestra reducida y estratificada (600/clase = 6000 imagenes, misma escala que
../07-reconocimiento-digitos/) en vez de las 70k completas -- entrenar 5 CNNs con mini-batch
sobre el dataset completo hubiera sido demasiado lento para una demostracion de la tecnica.

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
from lr_decay import DECAY_RATE
from weight_decay import WEIGHT_DECAY
from baseline_tf import BATCH_SIZE, EPOCHS_MAX, PATIENCE, build_model, graficar_confusion

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

N_POR_CLASE = 600
K_FOLDS = 5
SEED_MUESTREO = 42
SEED_SPLIT_FINAL = 42
SEED_KFOLD = 42
SEED_MODELO = 42


def muestrear_fashion_mnist(quiet=False):
    (x_tr, y_tr), (x_te, y_te) = tf.keras.datasets.fashion_mnist.load_data()
    X = np.concatenate([x_tr, x_te], axis=0)[..., np.newaxis].astype("float32") / 255.0
    Y = np.concatenate([y_tr, y_te], axis=0).astype("int64")

    rng = np.random.default_rng(SEED_MUESTREO)
    idx_muestra = []
    for clase in range(10):
        idx_clase = np.where(Y == clase)[0]
        rng.shuffle(idx_clase)
        idx_muestra.extend(idx_clase[:N_POR_CLASE])
    idx_muestra = np.array(idx_muestra)
    rng.shuffle(idx_muestra)

    if not quiet:
        print(f"Fashion-MNIST muestreado: {len(idx_muestra)} imagenes ({N_POR_CLASE}/clase)")
    return X[idx_muestra], Y[idx_muestra]


def crear_optimizador(steps_per_epoch):
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=LEARNING_RATE_ADAM, decay_steps=steps_per_epoch, decay_rate=DECAY_RATE, staircase=True)
    return tf.keras.optimizers.AdamW(learning_rate=lr_schedule, weight_decay=WEIGHT_DECAY)


def main(quiet=False) -> dict:
    X, Y = muestrear_fashion_mnist(quiet=quiet)

    idx_pool, idx_test_final = train_test_split(
        np.arange(len(X)), test_size=0.2, stratify=Y, random_state=SEED_SPLIT_FINAL)
    X_pool, Y_pool = X[idx_pool], Y[idx_pool]
    X_test_final, Y_test_final = X[idx_test_final], Y[idx_test_final]

    skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=SEED_KFOLD)
    accuracies_fold = []
    probs_por_modelo = []
    tiempo_total = 0.0

    for fold, (idx_train, idx_val) in enumerate(skf.split(X_pool, Y_pool)):
        X_train, X_val = X_pool[idx_train], X_pool[idx_val]
        Y_train, Y_val = Y_pool[idx_train], Y_pool[idx_val]
        steps_per_epoch = -(-len(X_train) // BATCH_SIZE)

        tf.keras.utils.set_random_seed(SEED_MODELO + fold)
        tf.config.experimental.enable_op_determinism()

        model = build_model()
        model.compile(optimizer=crear_optimizador(steps_per_epoch), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        t0 = time.time()
        model.fit(
            X_train, Y_train,
            validation_data=(X_val, Y_val),
            epochs=EPOCHS_MAX,
            batch_size=BATCH_SIZE,
            callbacks=[tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=PATIENCE, restore_best_weights=True)],
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

    matriz_confusion = np.zeros((10, 10), dtype=int)
    for real, pred in zip(Y_test_final, pred_ensemble):
        matriz_confusion[real, pred] += 1

    metrics = {
        "k_folds": K_FOLDS,
        "n_por_clase": N_POR_CLASE,
        "seed_split_final": SEED_SPLIT_FINAL,
        "seed_kfold": SEED_KFOLD,
        "seed_modelo": SEED_MODELO,
        "n_pool": len(X_pool),
        "n_test_final": len(X_test_final),
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
    plt.title(f"Ensembling + k-fold (k={K_FOLDS}) -- Fashion-MNIST (Keras)", fontweight="bold")
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
