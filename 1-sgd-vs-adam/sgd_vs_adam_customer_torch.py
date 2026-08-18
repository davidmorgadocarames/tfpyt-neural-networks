"""
SGD vs Adam (PyTorch) sobre el problema de tipos de clientes -- version PyTorch de
sgd_vs_adam_customer_tf.py, misma arquitectura (2 -> 5 LeakyReLU -> 3), mismos datos (120
clientes, split 60/20/20 estratificado) y misma fuente de datos (la funcion generar_datos() de
la version TF, importada aqui para que las dos versiones entrenen sobre exactamente los mismos
numeros). A diferencia de Keras, el bucle de entrenamiento, el early stopping y el checkpoint
del mejor punto de validacion se implementan a mano, igual que en sgd_vs_adam_dense_torch.py.

Corre en GPU si esta disponible. Guarda resultados en results_sgd_vs_adam/customer_torch/.

Uso: python sgd_vs_adam_customer_torch.py
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

from sgd_vs_adam_customer_tf import N_TRAIN_CLASE, N_VAL_CLASE, NOMBRES_CLASES, generar_datos

RESULTS_DIR = Path(__file__).parent / "results_sgd_vs_adam" / "customer_torch"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EPOCHS_MAX = 3000
PATIENCE = 200
MIN_DELTA = 1e-4  # ver el comentario en sgd_vs_adam_customer_tf.py -- sin Dropout, full-batch,
# determinista: hace falta min_delta para que el plateau dispare el early stopping.

VARIANTES = [
    ("sgd", lambda params: torch.optim.SGD(params, lr=0.5, momentum=0.0)),
    ("adam", lambda params: torch.optim.Adam(params, lr=0.02)),
]
COLORES_VARIANTES = {"sgd": "#4C72B0", "adam": "#55A868"}


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
    """Misma arquitectura que build_model() en sgd_vs_adam_customer_tf.py: 2 -> 5 LeakyReLU ->
    3 (logits; CrossEntropyLoss aplica log_softmax internamente)."""
    return nn.Sequential(
        nn.Linear(2, 5), nn.LeakyReLU(0.01),
        nn.Linear(5, 3),
    ).to(DEVICE)


def entrenar(nombre, model, optimizer, x_train, y_train, x_val, y_val, quiet=False):
    """Replica el algoritmo exacto de tf.keras.callbacks.EarlyStopping(min_delta=MIN_DELTA,
    patience=PATIENCE, restore_best_weights=True): 'best' (y los pesos guardados) solo se
    actualiza cuando la mejora supera min_delta -- no en cada minima mejora estricta -- y la
    cuenta de paciencia se reinicia solo entonces."""
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
                    print(f"  [{nombre}] early stopping en la epoca {epoch + 1}")
                break
    else:
        if not quiet:
            print(f"  [{nombre}] completo las {EPOCHS_MAX} epocas sin activar el early stopping")

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


def graficar_curva_comparativa(curvas_val, ruta_salida):
    plt.figure(figsize=(8, 4.5))
    for nombre, historial in curvas_val.items():
        plt.plot(historial, color=COLORES_VARIANTES[nombre], label=nombre.upper())
    plt.yscale("log")
    plt.title("SGD vs Adam (PyTorch): perdida de validacion por epoca (tipos de clientes)",
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
            print(f"\n=== Entrenando '{nombre}' (PyTorch, tipos-clientes, full-batch) ===")
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
        matriz_confusion = np.zeros((3, 3), dtype=int)
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
