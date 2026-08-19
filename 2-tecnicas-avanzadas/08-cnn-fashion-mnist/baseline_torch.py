"""
Baseline (PyTorch) de la CNN sobre Fashion-MNIST -- version PyTorch de baseline_tf.py, mismo
optimizador combinado (AdamW + LR-schedule + weight decay) y misma arquitectura
(Conv2D(32) -> BatchNorm -> ReLU -> MaxPool -> Conv2D(64) -> BatchNorm -> ReLU -> MaxPool ->
Flatten -> Dense(128) -> BatchNorm -> ReLU -> Dense(10)) con nn.BatchNorm2d/BatchNorm1d.

Uso: python baseline_torch.py
"""

import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf  # solo para cargar Fashion-MNIST -- mismos datos que la version TF
import torch
import torch.nn as nn

from adam import LEARNING_RATE_ADAM
from lr_decay import DECAY_RATE
from weight_decay import WEIGHT_DECAY

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

CLASS_NAMES = [
    "Camiseta", "Pantalon", "Jersey", "Vestido", "Abrigo",
    "Sandalia", "Camisa", "Zapatilla", "Bolso", "Bota",
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 128
EPOCHS_MAX = 30
PATIENCE = 5


def cargar_datos(seed_split, quiet=False):
    (x_train_full, y_train_full), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    x_train_full = x_train_full[:, np.newaxis, :, :].astype("float32") / 255.0
    x_test = x_test[:, np.newaxis, :, :].astype("float32") / 255.0
    y_train_full = y_train_full.astype("int64")
    y_test = y_test.astype("int64")

    rng = np.random.default_rng(seed_split)
    n = len(x_train_full)
    indices = rng.permutation(n)
    corte = int(n * 0.9)
    idx_train, idx_val = indices[:corte], indices[corte:]

    if not quiet:
        print(f"Fashion-MNIST completo: {len(idx_train)} train / {len(idx_val)} val / {len(x_test)} test "
              f"(device={DEVICE})")
    return (torch.from_numpy(x_train_full[idx_train]).to(DEVICE), torch.from_numpy(y_train_full[idx_train]).to(DEVICE),
            torch.from_numpy(x_train_full[idx_val]).to(DEVICE), torch.from_numpy(y_train_full[idx_val]).to(DEVICE),
            torch.from_numpy(x_test).to(DEVICE), torch.from_numpy(y_test).to(DEVICE))


def build_model() -> nn.Module:
    return nn.Sequential(
        nn.Conv2d(1, 32, 3, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(64 * 7 * 7, 128),
        nn.BatchNorm1d(128),
        nn.ReLU(),
        nn.Linear(128, 10),
    ).to(DEVICE)


def crear_optimizador_y_scheduler(model):
    """AdamW (weight decay desacoplado) + ExponentialLR (decae DECAY_RATE cada .step(), llamado
    una vez por epoca -- equivalente al steps_per_epoch de la version Keras)."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE_ADAM, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=DECAY_RATE)
    return optimizer, scheduler


def generar_batches(n, batch_size, generator):
    indices = torch.randperm(n, generator=generator)
    for inicio in range(0, n, batch_size):
        yield indices[inicio:inicio + batch_size]


def entrenar(model, optimizer, scheduler, x_train, y_train, x_val, y_val, seed_modelo, quiet=False):
    criterio = nn.CrossEntropyLoss()
    historial_loss_val = []
    mejor_loss_val = float("inf")
    mejor_epoca = None
    mejor_estado = None
    generator = torch.Generator(device="cpu").manual_seed(seed_modelo)
    n = x_train.shape[0]

    for epoch in range(EPOCHS_MAX):
        model.train()
        for idx_batch in generar_batches(n, BATCH_SIZE, generator):
            idx_batch = idx_batch.to(DEVICE)
            xb, yb = x_train[idx_batch], y_train[idx_batch]
            optimizer.zero_grad()
            loss = criterio(model(xb), yb)
            loss.backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            loss_val = criterio(model(x_val), y_val).item()
        historial_loss_val.append(loss_val)

        if loss_val < mejor_loss_val:
            mejor_loss_val = loss_val
            mejor_epoca = epoch
            mejor_estado = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch - mejor_epoca >= PATIENCE:
            if not quiet:
                print(f"  early stopping en la epoca {epoch + 1}")
            break
    else:
        if not quiet:
            print(f"  completo las {EPOCHS_MAX} epocas sin activar el early stopping")

    model.load_state_dict(mejor_estado)
    return historial_loss_val, mejor_epoca, mejor_loss_val


def graficar_confusion(matriz, accuracy, titulo, ruta):
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(matriz, cmap="Blues")
    ax.set_title(f"{titulo} (accuracy={accuracy:.2%})", fontweight="bold")
    ax.set_xlabel("Prediccion")
    ax.set_ylabel("Real")
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right")
    ax.set_yticklabels(CLASS_NAMES)
    for i in range(10):
        for j in range(10):
            valor = matriz[i, j]
            if valor > 0:
                ax.text(j, i, str(valor), ha="center", va="center",
                         color="white" if valor > matriz.max() / 2 else "black", fontsize=7)
    fig.colorbar(im, fraction=0.046)
    plt.tight_layout()
    plt.savefig(ruta, dpi=150)
    plt.close()


def graficar_curva(historial_loss_val, ruta_salida):
    plt.figure(figsize=(7, 4.5))
    plt.plot(historial_loss_val, color="#4C72B0")
    plt.yscale("log")
    plt.title("Baseline CNN (Adam+LRdecay+WeightDecay+BatchNorm): perdida de validacion", fontweight="bold")
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
    hist_val, mejor_epoca, mejor_loss_val = entrenar(
        model, optimizer, scheduler, x_train, y_train, x_val, y_val, seed_modelo, quiet=quiet)
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
    matriz_confusion = np.zeros((10, 10), dtype=int)
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
        "weight_decay": WEIGHT_DECAY,
        "batch_size": BATCH_SIZE,
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
