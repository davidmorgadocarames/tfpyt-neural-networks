"""
Baseline (PyTorch) del problema de tipos de clientes -- version PyTorch de baseline_tf.py,
mismo optimizador combinado (AdamW + LR-schedule + weight decay) y misma arquitectura.

Uso: python baseline_torch.py
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

from adam import LEARNING_RATE_ADAM
from lr_decay import DECAY_EVERY, DECAY_RATE
from weight_decay import WEIGHT_DECAY
from baseline_tf import N_TRAIN_CLASE, N_VAL_CLASE, NOMBRES_CLASES, generar_datos

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EPOCHS_MAX = 3000
PATIENCE = 200
MIN_DELTA = 1e-4


def cargar_datos(seed_split, quiet=False):
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
        print(f"Clientes: {len(X_train)} train / {len(X_val)} val / {len(X_test)} test (device={DEVICE})")
    return (torch.from_numpy(X_train).to(DEVICE), torch.from_numpy(Y_train).to(DEVICE),
            torch.from_numpy(X_val).to(DEVICE), torch.from_numpy(Y_val).to(DEVICE),
            torch.from_numpy(X_test).to(DEVICE), torch.from_numpy(Y_test).to(DEVICE))


def build_model() -> nn.Module:
    """Misma arquitectura que build_model() en baseline_tf.py: 2 -> 5 LeakyReLU -> 3 (logits;
    CrossEntropyLoss aplica log_softmax internamente)."""
    return nn.Sequential(
        nn.Linear(2, 5), nn.LeakyReLU(0.01),
        nn.Linear(5, 3),
    ).to(DEVICE)


def crear_optimizador_y_scheduler(model):
    """AdamW (weight decay desacoplado) + StepLR (decae DECAY_RATE cada DECAY_EVERY epocas,
    equivalente a la version Keras)."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE_ADAM, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=DECAY_EVERY, gamma=DECAY_RATE)
    return optimizer, scheduler


def entrenar(model, optimizer, scheduler, x_train, y_train, x_val, y_val, quiet=False):
    criterio = nn.CrossEntropyLoss()
    historial_loss_val = []
    mejor_loss_val = float("inf")
    mejor_epoca = None
    mejor_estado = None
    wait = 0

    for epoch in range(EPOCHS_MAX):
        model.train()
        optimizer.zero_grad()
        logits = model(x_train)
        loss = criterio(logits, y_train)
        loss.backward()
        optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            loss_val = criterio(model(x_val), y_val).item()
        historial_loss_val.append(loss_val)

        if loss_val < mejor_loss_val - MIN_DELTA:
            mejor_loss_val = loss_val
            mejor_epoca = epoch
            mejor_estado = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= PATIENCE:
                if not quiet:
                    print(f"  early stopping en la epoca {epoch + 1}")
                break
    else:
        if not quiet:
            print(f"  completo las {EPOCHS_MAX} epocas sin activar el early stopping")

    model.load_state_dict(mejor_estado)
    return historial_loss_val, mejor_epoca, mejor_loss_val


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


def graficar_curva(historial_loss_val, ruta_salida):
    plt.figure(figsize=(7, 4.5))
    plt.plot(historial_loss_val, color="#4C72B0")
    plt.yscale("log")
    plt.title("Baseline (Adam+LRdecay+WeightDecay): perdida de validacion por epoca", fontweight="bold")
    plt.xlabel("Epoca")
    plt.ylabel("Perdida de validacion (escala log)")
    plt.grid(True, which="both", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(ruta_salida, dpi=150)
    plt.close()


def main(seed_split=42, seed_modelo=42, quiet=False, guardar_graficas=True) -> dict:
    x_train, y_train, x_val, y_val, x_test, y_test = cargar_datos(seed_split, quiet=quiet)

    torch.manual_seed(seed_modelo)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_modelo)

    model = build_model()
    optimizer, scheduler = crear_optimizador_y_scheduler(model)

    t0 = time.time()
    hist_val, mejor_epoca, mejor_loss_val = entrenar(model, optimizer, scheduler, x_train, y_train, x_val, y_val, quiet=quiet)
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    tiempo_computacion_segundos = time.time() - t0
    epocas_entrenadas = len(hist_val)

    model.eval()
    with torch.no_grad():
        logits_test = model(x_test)
        criterio = nn.CrossEntropyLoss()
        loss_test = criterio(logits_test, y_test).item()
        pred_test = torch.argmax(logits_test, dim=1)
        accuracy_test = (pred_test == y_test).float().mean().item()

    pred_np, y_np = pred_test.cpu().numpy(), y_test.cpu().numpy()
    matriz_confusion = np.zeros((3, 3), dtype=int)
    for real, pred in zip(y_np, pred_np):
        matriz_confusion[real, pred] += 1

    if not quiet:
        print(f"accuracy test: {accuracy_test:.4f} -- epocas: {epocas_entrenadas} "
              f"(mejor val en {mejor_epoca + 1}) -- {tiempo_computacion_segundos:.1f}s")

    metrics = {
        "seed_split": seed_split,
        "seed_modelo": seed_modelo,
        "device": str(DEVICE),
        "learning_rate_adam": LEARNING_RATE_ADAM,
        "decay_rate": DECAY_RATE,
        "decay_every": DECAY_EVERY,
        "weight_decay": WEIGHT_DECAY,
        "epochs_max_configuradas": EPOCHS_MAX,
        "epochs_entrenadas": epocas_entrenadas,
        "epoca_mejor_val": mejor_epoca + 1,
        "loss_val_final": float(mejor_loss_val),
        "accuracy_test": float(accuracy_test),
        "loss_test": float(loss_test),
        "tiempo_computacion_segundos": tiempo_computacion_segundos,
        "n_train": len(x_train), "n_val": len(x_val), "n_test": len(x_test),
        "matriz_confusion": matriz_confusion.tolist(),
        "historial_loss_val": hist_val,
    }

    if not guardar_graficas:
        return metrics

    (RESULTS_DIR / "baseline_torch_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    graficar_confusion(matriz_confusion, accuracy_test, "Matriz de confusion test -- baseline",
                        RESULTS_DIR / "baseline_torch_confusion.png")
    graficar_curva(hist_val, RESULTS_DIR / "baseline_torch_curva.png")

    if not quiet:
        print(f"Resultados guardados en {RESULTS_DIR}")

    return metrics


if __name__ == "__main__":
    main()
