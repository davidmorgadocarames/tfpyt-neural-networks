"""
Ensembling + validacion cruzada combinados (PyTorch) sobre temperatura dia/noche -- version
PyTorch de ensembling_kfold_tf.py (mismo uso de TimeSeriesSplit).

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
from sklearn.model_selection import TimeSeriesSplit

from baseline_torch import DEVICE, EPOCHS_MAX, MIN_DELTA, PATIENCE, build_model, crear_optimizador_y_scheduler
from baseline_tf import VENTANA, generar_serie

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

K_FOLDS = 5
SEED_MODELO = 42


def entrenar(model, optimizer, scheduler, x_train, y_train, x_val, y_val):
    criterio = nn.MSELoss()
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
    x_test_final_t = torch.from_numpy(X_test_final).to(DEVICE)

    tscv = TimeSeriesSplit(n_splits=K_FOLDS)
    mae_fold = []
    preds_por_modelo = []
    tiempo_total = 0.0

    for fold, (idx_train, idx_val) in enumerate(tscv.split(X_pool)):
        x_train_t = torch.from_numpy(X_pool[idx_train]).to(DEVICE)
        y_train_t = torch.from_numpy(Y_pool[idx_train]).to(DEVICE)
        x_val_t = torch.from_numpy(X_pool[idx_val]).to(DEVICE)
        y_val_t = torch.from_numpy(Y_pool[idx_val]).to(DEVICE)

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
            pred_val_norm = model(x_val_t).cpu().numpy()
            pred_test_norm = model(x_test_final_t).cpu().numpy()

        y_val_c = Y_pool[idx_val].flatten() * (t_max - t_min) + t_min
        pred_val_c = pred_val_norm.flatten() * (t_max - t_min) + t_min
        mae_val = float(np.mean(np.abs(pred_val_c - y_val_c)))
        mae_fold.append(mae_val)
        preds_por_modelo.append(pred_test_norm.flatten())

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
        "device": str(DEVICE),
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
    (RESULTS_DIR / "ensembling_kfold_torch_metrics.json").write_text(
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
    plt.title(f"Ensembling + TimeSeriesSplit (k={K_FOLDS}) -- temperatura (PyTorch)", fontweight="bold")
    plt.legend(fontsize=8)
    plt.grid(True, axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "ensembling_kfold_torch_mae.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 4.5))
    plt.plot(Y_test_final_c, label="Temperatura real", color="#4C72B0")
    plt.plot(pred_ensemble_c, label="Temperatura predicha (ensemble)", color="#C44E52", alpha=0.8)
    plt.xlabel("Muestras de test_final (orden cronologico)")
    plt.ylabel("Temperatura (°C)")
    plt.title("Ensemble (PyTorch): temperatura real vs predicha (test_final)", fontweight="bold")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "ensembling_kfold_torch_prediccion.png", dpi=150)
    plt.close()

    if not quiet:
        print(f"\nMAE medio por fold (val): {media_fold:.3f} +/- {desviacion_fold:.3f} C")
        print(f"MAE medio individual (test_final): {metrics['media_individual_test_final_celsius']:.3f} C")
        print(f"MAE ensemble (test_final): {mae_ensemble_test:.3f} C")
        print(f"Resultados guardados en {RESULTS_DIR}")

    return metrics


if __name__ == "__main__":
    main()
