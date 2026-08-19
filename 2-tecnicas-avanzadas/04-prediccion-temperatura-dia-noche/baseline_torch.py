"""
Baseline (PyTorch) del problema de prediccion de temperatura dia/noche -- version PyTorch de
baseline_tf.py, mismo optimizador combinado (AdamW + LR-schedule + weight decay) y misma
arquitectura LSTM.

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
from baseline_tf import VENTANA, generar_serie

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EPOCHS_MAX = 3000
PATIENCE = 200
MIN_DELTA = 1e-6


def cargar_datos(quiet=False):
    dias, temperatura_c = generar_serie()

    n = len(temperatura_c) - VENTANA
    split_train = int(n * 0.6)
    split_val = int(n * 0.8)
    raw_fin_train = split_train + VENTANA

    t_min, t_max = np.min(temperatura_c[:raw_fin_train]), np.max(temperatura_c[:raw_fin_train])
    serie_norm = (temperatura_c - t_min) / (t_max - t_min)

    X_lista, Y_lista = [], []
    for i in range(len(serie_norm) - VENTANA):
        X_lista.append(serie_norm[i : i + VENTANA])
        Y_lista.append(serie_norm[i + VENTANA])
    X = np.array(X_lista, dtype="float32").reshape(-1, VENTANA, 1)
    Y = np.array(Y_lista, dtype="float32").reshape(-1, 1)

    X_train, X_val, X_test = X[:split_train], X[split_train:split_val], X[split_val:]
    Y_train, Y_val, Y_test = Y[:split_train], Y[split_train:split_val], Y[split_val:]

    if not quiet:
        print(f"Temperatura: {len(X_train)} train / {len(X_val)} val / {len(X_test)} test "
              f"(ventana={VENTANA}h, device={DEVICE})")
    return (torch.from_numpy(X_train).to(DEVICE), torch.from_numpy(Y_train).to(DEVICE),
            torch.from_numpy(X_val).to(DEVICE), torch.from_numpy(Y_val).to(DEVICE),
            torch.from_numpy(X_test).to(DEVICE), torch.from_numpy(Y_test).to(DEVICE),
            t_min, t_max, dias, split_val)


class ModeloLSTM(nn.Module):
    """LSTM(32) -> Dense(16, relu) -> Dense(1), misma arquitectura que build_model() en
    baseline_tf.py."""

    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=32, batch_first=True)
        self.densa1 = nn.Linear(32, 16)
        self.relu = nn.ReLU()
        self.densa2 = nn.Linear(16, 1)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        h_final = h_n[-1]
        return self.densa2(self.relu(self.densa1(h_final)))


def build_model() -> nn.Module:
    return ModeloLSTM().to(DEVICE)


def crear_optimizador_y_scheduler(model):
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE_ADAM, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=DECAY_EVERY, gamma=DECAY_RATE)
    return optimizer, scheduler


def entrenar(model, optimizer, scheduler, x_train, y_train, x_val, y_val, quiet=False):
    criterio = nn.MSELoss()
    historial_loss_val = []
    mejor_loss_val = float("inf")
    mejor_epoca = None
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


def graficar_curva(historial_loss_val, ruta_salida):
    plt.figure(figsize=(7, 4.5))
    plt.plot(historial_loss_val, color="#4C72B0")
    plt.yscale("log")
    plt.title("Baseline LSTM (Adam+LRdecay+WeightDecay): MSE de validacion por epoca", fontweight="bold")
    plt.xlabel("Epoca")
    plt.ylabel("MSE de validacion, normalizado (escala log)")
    plt.grid(True, which="both", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(ruta_salida, dpi=150)
    plt.close()


def graficar_prediccion(y_test_c, y_pred_c, horas_test, ruta_salida):
    plt.figure(figsize=(8, 4.5))
    plt.plot(horas_test, y_test_c, label="Temperatura real", color="#4C72B0")
    plt.plot(horas_test, y_pred_c, label="Temperatura predicha", color="#C44E52", alpha=0.8)
    plt.xlabel("Horas dentro del tramo de test")
    plt.ylabel("Temperatura (°C)")
    plt.title("Baseline LSTM (PyTorch): temperatura real vs predicha (test)", fontweight="bold")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(ruta_salida, dpi=150)
    plt.close()


def main(seed_split=42, seed_modelo=42, quiet=False, guardar_graficas=True) -> dict:
    x_train, y_train, x_val, y_val, x_test, y_test, t_min, t_max, dias, split_val = cargar_datos(quiet=quiet)

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
        y_pred_norm = model(x_test).cpu().numpy()
        criterio = nn.MSELoss()
        loss_test = criterio(model(x_test), y_test).item()

    y_test_c = y_test.cpu().numpy().flatten() * (t_max - t_min) + t_min
    y_pred_c = y_pred_norm.flatten() * (t_max - t_min) + t_min
    mae_test_celsius = float(np.mean(np.abs(y_pred_c - y_test_c)))
    rmse_test_celsius = float(np.sqrt(np.mean((y_pred_c - y_test_c) ** 2)))

    if not quiet:
        print(f"MAE test: {mae_test_celsius:.3f} C -- epocas: {epocas_entrenadas} "
              f"(mejor val en {mejor_epoca + 1}) -- {tiempo_computacion_segundos:.1f}s")

    horas_test = (dias[VENTANA + split_val : VENTANA + split_val + len(y_test)]
                  - dias[VENTANA + split_val]) * 24

    metrics = {
        "seed_modelo": seed_modelo,
        "device": str(DEVICE),
        "ventana_horas": VENTANA,
        "learning_rate_adam": LEARNING_RATE_ADAM,
        "decay_rate": DECAY_RATE,
        "decay_every": DECAY_EVERY,
        "weight_decay": WEIGHT_DECAY,
        "epochs_max_configuradas": EPOCHS_MAX,
        "epochs_entrenadas": epocas_entrenadas,
        "epoca_mejor_val": mejor_epoca + 1,
        "loss_val_final": float(mejor_loss_val),
        "mae_test_celsius": mae_test_celsius,
        "rmse_test_celsius": rmse_test_celsius,
        "tiempo_computacion_segundos": tiempo_computacion_segundos,
        "n_train": len(x_train), "n_val": len(x_val), "n_test": len(x_test),
        "historial_loss_val": hist_val,
        "y_test_celsius": y_test_c.tolist(),
        "y_pred_celsius": y_pred_c.tolist(),
        "horas_test": horas_test.tolist(),
    }

    if not guardar_graficas:
        return metrics

    (RESULTS_DIR / "baseline_torch_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    graficar_curva(hist_val, RESULTS_DIR / "baseline_torch_curva.png")
    graficar_prediccion(y_test_c, y_pred_c, horas_test, RESULTS_DIR / "baseline_torch_prediccion.png")

    if not quiet:
        print(f"Resultados guardados en {RESULTS_DIR}")

    return metrics


if __name__ == "__main__":
    main()
