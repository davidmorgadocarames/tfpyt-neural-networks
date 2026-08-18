"""
SGD vs Adam (PyTorch) sobre MNIST completo, mini-batch (batch_size=128, ~422 actualizaciones
de pesos por epoca) -- version PyTorch de sgd_vs_adam_dense_tf_full.py. Todo el dataset se
mueve a la GPU una sola vez (~180 MB) y los mini-batches se indexan alli directamente, sin
transferencia CPU->GPU en cada batch. Guarda resultados en
results_sgd_vs_adam/dense_torch_minibatch/.

Uso: python sgd_vs_adam_dense_torch_full.py
"""

import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf  # solo para cargar MNIST -- mismos datos que la version TF
import torch
import torch.nn as nn

RESULTS_DIR = Path(__file__).parent / "results_sgd_vs_adam" / "dense_torch_minibatch"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 128
EPOCHS_MAX = 30
PATIENCE = 5

VARIANTES = [
    ("sgd", lambda params: torch.optim.SGD(params, lr=0.1, momentum=0.0)),
    ("adam", lambda params: torch.optim.Adam(params, lr=0.001)),
]
COLORES_VARIANTES = {"sgd": "#4C72B0", "adam": "#55A868"}


def cargar_datos(seed_split, quiet=False):
    (x_train_full, y_train_full), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train_full = x_train_full.reshape(-1, 28 * 28).astype("float32") / 255.0
    x_test = x_test.reshape(-1, 28 * 28).astype("float32") / 255.0
    y_train_full = y_train_full.astype("int64")
    y_test = y_test.astype("int64")

    rng = np.random.default_rng(seed_split)
    n = len(x_train_full)
    indices = rng.permutation(n)
    corte = int(n * 0.9)
    idx_train, idx_val = indices[:corte], indices[corte:]

    if not quiet:
        print(f"MNIST completo: {len(idx_train)} train / {len(idx_val)} val / {len(x_test)} test "
              f"(device={DEVICE})")
    return (torch.from_numpy(x_train_full[idx_train]).to(DEVICE), torch.from_numpy(y_train_full[idx_train]).to(DEVICE),
            torch.from_numpy(x_train_full[idx_val]).to(DEVICE), torch.from_numpy(y_train_full[idx_val]).to(DEVICE),
            torch.from_numpy(x_test).to(DEVICE), torch.from_numpy(y_test).to(DEVICE))


def build_model() -> nn.Module:
    return nn.Sequential(
        nn.Linear(28 * 28, 128), nn.LeakyReLU(0.01), nn.Dropout(0.3),
        nn.Linear(128, 64), nn.LeakyReLU(0.01), nn.Dropout(0.3),
        nn.Linear(64, 10),
    ).to(DEVICE)


def generar_batches(n, batch_size, generator):
    """randperm en CPU (el generator es CPU) -- los indices se mueven a DEVICE al usarlos,
    barato porque son enteros, no las imagenes."""
    indices = torch.randperm(n, generator=generator)
    for inicio in range(0, n, batch_size):
        yield indices[inicio:inicio + batch_size]


def entrenar(nombre, model, optimizer, x_train, y_train, x_val, y_val, seed_modelo, quiet=False):
    criterio = nn.CrossEntropyLoss()
    historial_loss_val = []
    mejor_loss_val = float("inf")
    mejor_epoca = None
    mejor_estado = None
    generator = torch.Generator(device="cpu").manual_seed(seed_modelo)
    n = x_train.shape[0]

    for epoch in range(EPOCHS_MAX):
        model.train()
        loss_train_acumulada, n_vistas = 0.0, 0
        for idx_batch in generar_batches(n, BATCH_SIZE, generator):
            idx_batch = idx_batch.to(DEVICE)
            xb, yb = x_train[idx_batch], y_train[idx_batch]
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterio(logits, yb)
            loss.backward()
            optimizer.step()
            loss_train_acumulada += loss.item() * xb.shape[0]
            n_vistas += xb.shape[0]

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
                print(f"  [{nombre}] early stopping en la epoca {epoch + 1}")
            break
    else:
        if not quiet:
            print(f"  [{nombre}] completo las {EPOCHS_MAX} epocas sin activar el early stopping")

    model.load_state_dict(mejor_estado)
    return historial_loss_val, mejor_epoca, mejor_loss_val


def graficar_confusion(matriz, accuracy, titulo, ruta):
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(matriz, cmap="Blues")
    ax.set_title(f"{titulo} (accuracy={accuracy:.2%})", fontweight="bold")
    ax.set_xlabel("Prediccion")
    ax.set_ylabel("Real")
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    for i in range(10):
        for j in range(10):
            valor = matriz[i, j]
            if valor > 0:
                ax.text(j, i, str(valor), ha="center", va="center",
                         color="white" if valor > matriz.max() / 2 else "black", fontsize=8)
    fig.colorbar(im, fraction=0.046)
    plt.tight_layout()
    plt.savefig(ruta, dpi=150)
    plt.close()


def graficar_curva_comparativa(curvas_val, ruta_salida):
    plt.figure(figsize=(8, 4.5))
    for nombre, historial in curvas_val.items():
        plt.plot(historial, color=COLORES_VARIANTES[nombre], label=nombre.upper())
    plt.yscale("log")
    plt.title("SGD vs Adam (PyTorch): perdida de validacion por epoca (MNIST completo, mini-batch)",
               fontweight="bold")
    plt.xlabel("Epoca")
    plt.ylabel("Perdida de validacion (escala log)")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(ruta_salida, dpi=150)
    plt.close()


def main(seed_split=42, seed_modelo=42, quiet=False, guardar_graficas=True) -> dict:
    x_train, y_train, x_val, y_val, x_test, y_test = cargar_datos(seed_split, quiet=quiet)

    resultados = {}
    curvas_val = {}
    for nombre, crear_optimizador in VARIANTES:
        if not quiet:
            print(f"\n=== Entrenando '{nombre}' (PyTorch, mini-batch) ===")
        torch.manual_seed(seed_modelo)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed_modelo)

        model = build_model()
        optimizer = crear_optimizador(model.parameters())

        t0 = time.time()
        hist_val, mejor_epoca, mejor_loss_val = entrenar(
            nombre, model, optimizer, x_train, y_train, x_val, y_val, seed_modelo, quiet=quiet
        )
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        tiempo = time.time() - t0
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
            print(f"  [{nombre}] accuracy test: {accuracy_test:.4f} -- epocas: {epocas_entrenadas} "
                  f"(mejor val en {mejor_epoca + 1}) -- {tiempo:.1f}s")

        if guardar_graficas:
            graficar_confusion(matriz_confusion, accuracy_test, f"Matriz de confusion test -- {nombre.upper()}",
                                RESULTS_DIR / f"confusion_matrix_{nombre}.png")

        resultados[nombre] = {
            "epochs_entrenadas": epocas_entrenadas,
            "epoca_mejor_val": mejor_epoca + 1,
            "loss_val_final": float(mejor_loss_val),
            "accuracy_test": float(accuracy_test),
            "loss_test": float(loss_test),
            "tiempo_segundos": tiempo,
            "matriz_confusion": matriz_confusion.tolist(),
            "historial_loss_val": hist_val,
        }
        curvas_val[nombre] = hist_val

    metrics = {
        "seed_split": seed_split,
        "seed_modelo": seed_modelo,
        "device": str(DEVICE),
        "batch_size": BATCH_SIZE,
        "epochs_max_configuradas": EPOCHS_MAX,
        "n_train": len(x_train), "n_val": len(x_val), "n_test": len(x_test),
        "resultados": resultados,
    }

    if not guardar_graficas:
        return metrics

    (RESULTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    graficar_curva_comparativa(curvas_val, RESULTS_DIR / "learning_curve_comparativa.png")

    if not quiet:
        print("\n--- Resumen ---")
        for nombre, datos in resultados.items():
            print(f"{nombre:6s} accuracy={datos['accuracy_test']:.4f}  epocas={datos['epochs_entrenadas']}  "
                  f"tiempo={datos['tiempo_segundos']:.1f}s")
        print(f"Resultados guardados en {RESULTS_DIR}")

    return metrics


if __name__ == "__main__":
    main()
