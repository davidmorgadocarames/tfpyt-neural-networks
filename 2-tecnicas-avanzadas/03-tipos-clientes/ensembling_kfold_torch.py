"""
Ensembling + validacion cruzada k-fold combinados (PyTorch) -- version PyTorch de
ensembling_kfold_tf.py, cada uno de los K modelos usa la config combinada de
baseline_torch.py (AdamW + LR-schedule + weight decay).

Uso: python ensembling_kfold_torch.py
"""

import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold, train_test_split

from baseline_torch import DEVICE, EPOCHS_MAX, MIN_DELTA, PATIENCE, build_model, crear_optimizador_y_scheduler, graficar_confusion
from baseline_tf import generar_datos

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

K_FOLDS = 5
SEED_SPLIT_FINAL = 42
SEED_KFOLD = 42
SEED_MODELO = 42


def entrenar(model, optimizer, scheduler, x_train, y_train, x_val, y_val):
    criterio = nn.CrossEntropyLoss()
    mejor_loss_val = float("inf")
    mejor_estado = None
    wait = 0

    for epoch in range(EPOCHS_MAX):
        model.train()
        optimizer.zero_grad()
        loss = criterio(model(x_train), y_train)
        loss.backward()
        optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            loss_val = criterio(model(x_val), y_val).item()

        if loss_val < mejor_loss_val - MIN_DELTA:
            mejor_loss_val = loss_val
            mejor_estado = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= PATIENCE:
                break

    model.load_state_dict(mejor_estado)
    return mejor_loss_val


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
        Y_train_np, Y_val_np = Y_pool[idx_train], Y_pool[idx_val]

        X_min, X_max = X_train_raw.min(axis=0), X_train_raw.max(axis=0)
        X_train = ((X_train_raw - X_min) / (X_max - X_min)).astype("float32")
        X_val = ((X_val_raw - X_min) / (X_max - X_min)).astype("float32")
        X_test_final = ((X_test_final_raw - X_min) / (X_max - X_min)).astype("float32")

        x_train_t = torch.from_numpy(X_train).to(DEVICE)
        y_train_t = torch.from_numpy(Y_train_np).to(DEVICE)
        x_val_t = torch.from_numpy(X_val).to(DEVICE)
        y_val_t = torch.from_numpy(Y_val_np).to(DEVICE)
        x_test_final_t = torch.from_numpy(X_test_final).to(DEVICE)

        torch.manual_seed(SEED_MODELO + fold)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(SEED_MODELO + fold)

        model = build_model()
        optimizer, scheduler = crear_optimizador_y_scheduler(model)

        t0 = time.time()
        entrenar(model, optimizer, scheduler, x_train_t, y_train_t, x_val_t, y_val_t)
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        tiempo_total += time.time() - t0

        model.eval()
        with torch.no_grad():
            pred_val = torch.argmax(model(x_val_t), dim=1)
            acc_val = (pred_val == y_val_t).float().mean().item()
            probs_test = torch.softmax(model(x_test_final_t), dim=1).cpu().numpy()

        accuracies_fold.append(float(acc_val))
        probs_por_modelo.append(probs_test)

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
        "device": str(DEVICE),
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
    (RESULTS_DIR / "ensembling_kfold_torch_metrics.json").write_text(
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
    plt.title(f"Ensembling + k-fold (k={K_FOLDS}) -- tipos de clientes (PyTorch)", fontweight="bold")
    plt.legend(fontsize=8)
    plt.grid(True, axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "ensembling_kfold_torch_accuracy.png", dpi=150)
    plt.close()

    graficar_confusion(matriz_confusion, accuracy_ensemble, "Matriz de confusion -- ensemble sobre test_final",
                        RESULTS_DIR / "ensembling_kfold_torch_confusion.png")

    if not quiet:
        print(f"\nAccuracy media por fold (val): {media_fold:.4f} +/- {desviacion_fold:.4f}")
        print(f"Accuracy media individual (test_final): {metrics['media_individual_test_final']:.4f}")
        print(f"Accuracy ensemble (test_final): {accuracy_ensemble:.4f}")
        print(f"Resultados guardados en {RESULTS_DIR}")

    return metrics


if __name__ == "__main__":
    main()
