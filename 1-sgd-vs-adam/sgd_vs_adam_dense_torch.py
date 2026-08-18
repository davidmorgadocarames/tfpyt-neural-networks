"""
SGD vs Adam (PyTorch) sobre una muestra reducida de MNIST, full-batch (batch = todo el train,
1 actualizacion de pesos por epoca) -- version PyTorch de sgd_vs_adam_dense_tf.py, misma
arquitectura, mismos tamanos de muestra (1200/300/300, 120/30/30 por digito) y misma fuente de
datos (tf.keras.datasets, para que TF y PyTorch entrenen exactamente sobre las mismas imagenes
y la comparacion de framework sea limpia). A diferencia de Keras, PyTorch no tiene fit(): el
bucle de entrenamiento, el early stopping y el checkpoint del mejor punto de validacion se
implementan a mano, igual que en digit_classifier.py de RRNN (ver ese README para el porque).

Corre en GPU si esta disponible (torch.cuda.is_available()) -- en esta maquina, una RTX 4060 --
y lo reporta con honestidad si no. Guarda resultados en
results_sgd_vs_adam/dense_torch_fullbatch/.

Uso: python sgd_vs_adam_dense_torch.py
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

RESULTS_DIR = Path(__file__).parent / "results_sgd_vs_adam" / "dense_torch_fullbatch"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

N_TRAIN, N_VAL, N_TEST = 1200, 300, 300
EPOCHS_MAX = 600
PATIENCE = 60

VARIANTES = [
    ("sgd", lambda params: torch.optim.SGD(params, lr=0.5, momentum=0.0)),
    ("adam", lambda params: torch.optim.Adam(params, lr=0.005)),
]
COLORES_VARIANTES = {"sgd": "#4C72B0", "adam": "#55A868"}


def cargar_datos(seed_split, quiet=False):
    (x_tr, y_tr), (x_te, y_te) = tf.keras.datasets.mnist.load_data()
    X = np.concatenate([x_tr, x_te], axis=0).reshape(-1, 28 * 28).astype("float32") / 255.0
    Y = np.concatenate([y_tr, y_te], axis=0).astype("int64")

    rng = np.random.default_rng(seed_split)
    idx_train, idx_val, idx_test = [], [], []
    for digito in range(10):
        idx_digito = np.where(Y == digito)[0]
        rng.shuffle(idx_digito)
        c_train = N_TRAIN // 10
        c_val = c_train + N_VAL // 10
        c_test = c_val + N_TEST // 10
        idx_train.extend(idx_digito[:c_train])
        idx_val.extend(idx_digito[c_train:c_val])
        idx_test.extend(idx_digito[c_val:c_test])
    idx_train, idx_val, idx_test = np.array(idx_train), np.array(idx_val), np.array(idx_test)
    rng.shuffle(idx_train)
    rng.shuffle(idx_val)
    rng.shuffle(idx_test)

    if not quiet:
        print(f"MNIST muestreado: {len(idx_train)} train / {len(idx_val)} val / {len(idx_test)} test "
              f"(device={DEVICE})")
    return (torch.from_numpy(X[idx_train]).to(DEVICE), torch.from_numpy(Y[idx_train]).to(DEVICE),
            torch.from_numpy(X[idx_val]).to(DEVICE), torch.from_numpy(Y[idx_val]).to(DEVICE),
            torch.from_numpy(X[idx_test]).to(DEVICE), torch.from_numpy(Y[idx_test]).to(DEVICE))


def build_model() -> nn.Module:
    """Misma arquitectura que build_model() en sgd_vs_adam_dense_tf.py: 784 -> 128 LeakyReLU ->
    Dropout(0.3) -> 64 LeakyReLU -> Dropout(0.3) -> 10 (logits; CrossEntropyLoss aplica
    log_softmax internamente, no hace falta una capa Softmax explicita)."""
    return nn.Sequential(
        nn.Linear(28 * 28, 128), nn.LeakyReLU(0.01), nn.Dropout(0.3),
        nn.Linear(128, 64), nn.LeakyReLU(0.01), nn.Dropout(0.3),
        nn.Linear(64, 10),
    ).to(DEVICE)


def entrenar(nombre, model, optimizer, x_train, y_train, x_val, y_val, quiet=False):
    criterio = nn.CrossEntropyLoss()
    historial_loss_val = []
    mejor_loss_val = float("inf")
    mejor_epoca = None
    mejor_estado = None

    for epoch in range(EPOCHS_MAX):
        model.train()
        optimizer.zero_grad()
        logits = model(x_train)
        loss = criterio(logits, y_train)
        loss.backward()
        optimizer.step()

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
    plt.title("SGD vs Adam (PyTorch): perdida de validacion por epoca (muestra reducida, full-batch)",
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
            print(f"\n=== Entrenando '{nombre}' (PyTorch, full-batch) ===")
        torch.manual_seed(seed_modelo)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed_modelo)

        model = build_model()
        optimizer = crear_optimizador(model.parameters())

        t0 = time.time()
        hist_val, mejor_epoca, mejor_loss_val = entrenar(
            nombre, model, optimizer, x_train, y_train, x_val, y_val, quiet=quiet
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
