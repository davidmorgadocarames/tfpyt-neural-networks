"""
Test de permutacion (PyTorch) -- version PyTorch de permutation_test_tf.py, misma metodologia
y config combinada (AdamW+LRdecay+WeightDecay+LSTM) de baseline_torch.py.

Uso:
    python permutation_test_torch.py
    python permutation_test_torch.py --n 20
"""

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from baseline_torch import DEVICE, EPOCHS_MAX, MIN_DELTA, PATIENCE, build_model, crear_optimizador_y_scheduler
from baseline_tf import AMPLITUD_C, N_DIAS, PUNTOS_POR_DIA, RUIDO_STD_C, SEED_DATOS, TEMP_MEDIA_C, VENTANA
from permutation_test_tf import N_PERMUTACIONES_DEFAULT, SEED_MODELO_FIJO, SEED_PERMUTACION, generar_serie_real

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def entrenar_evaluar(serie_temperatura, seed_modelo):
    n_puntos = len(serie_temperatura)
    n = n_puntos - VENTANA
    split_train = int(n * 0.6)
    split_val = int(n * 0.8)
    raw_fin_train = split_train + VENTANA

    t_min = np.min(serie_temperatura[:raw_fin_train])
    t_max = np.max(serie_temperatura[:raw_fin_train])
    serie_norm = (serie_temperatura - t_min) / (t_max - t_min)

    X_lista, Y_lista = [], []
    for i in range(len(serie_norm) - VENTANA):
        X_lista.append(serie_norm[i : i + VENTANA])
        Y_lista.append(serie_norm[i + VENTANA])
    X = np.array(X_lista, dtype="float32").reshape(-1, VENTANA, 1)
    Y = np.array(Y_lista, dtype="float32").reshape(-1, 1)

    x_train = torch.from_numpy(X[:split_train]).to(DEVICE)
    y_train = torch.from_numpy(Y[:split_train]).to(DEVICE)
    x_val = torch.from_numpy(X[split_train:split_val]).to(DEVICE)
    y_val = torch.from_numpy(Y[split_train:split_val]).to(DEVICE)
    x_test = torch.from_numpy(X[split_val:]).to(DEVICE)
    y_test_np = Y[split_val:]

    torch.manual_seed(seed_modelo)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_modelo)

    model = build_model()
    optimizer, scheduler = crear_optimizador_y_scheduler(model)
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

    model.eval()
    with torch.no_grad():
        y_pred_norm = model(x_test).cpu().numpy()

    y_test_c = y_test_np.flatten() * (t_max - t_min) + t_min
    y_pred_c = y_pred_norm.flatten() * (t_max - t_min) + t_min
    return float(np.mean(np.abs(y_pred_c - y_test_c)))


def graficar_histograma(mae_permutados, mae_real, p_valor, ruta_salida):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(mae_permutados, bins=min(40, len(mae_permutados)), color="#4C72B0", edgecolor="black",
            alpha=0.85, label=f"MAE con datos barajados (N={len(mae_permutados)})")
    ax.axvline(mae_real, color="#C44E52", linestyle="--", linewidth=2,
               label=f"MAE con datos reales = {mae_real:.2f} °C")
    cota_p = 1 / (len(mae_permutados) + 1)
    texto_p = f"p < {cota_p:.3g}" if p_valor <= cota_p else f"p = {p_valor:.3g}"
    ax.text(mae_real, ax.get_ylim()[1] * 0.92, f"  {texto_p}", color="#C44E52", fontsize=9, va="top")
    ax.set_xlabel("MAE en test (°C)")
    ax.set_ylabel("Nº de barajados")
    ax.set_title("Test de permutacion (PyTorch): ¿es real el ciclo dia/noche aprendido?",
                  fontweight="bold", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    plt.tight_layout()
    plt.savefig(ruta_salida, dpi=150)
    plt.close()


def main(n_permutaciones=N_PERMUTACIONES_DEFAULT):
    serie_real = generar_serie_real()

    print(f"Entrenando sobre datos reales (seed_modelo={SEED_MODELO_FIJO}, device={DEVICE})...")
    t0 = time.time()
    mae_real = entrenar_evaluar(serie_real, SEED_MODELO_FIJO)
    print(f"MAE en test, datos reales: {mae_real:.4f} °C ({time.time() - t0:.0f}s)")

    rng_permutacion = np.random.default_rng(SEED_PERMUTACION)
    mae_permutados = []
    t0 = time.time()
    for i in range(n_permutaciones):
        serie_barajada = rng_permutacion.permutation(serie_real)
        mae = entrenar_evaluar(serie_barajada, SEED_MODELO_FIJO)
        mae_permutados.append(mae)
        if (i + 1) % 10 == 0 or (i + 1) == n_permutaciones:
            print(f"  {i + 1}/{n_permutaciones} barajados -- {time.time() - t0:.0f}s acumulados", flush=True)

    mae_permutados = np.array(mae_permutados)
    n_iguala_o_mejora = int(np.sum(mae_permutados <= mae_real))
    p_valor = (n_iguala_o_mejora + 1) / (n_permutaciones + 1)

    resumen = {
        "device": str(DEVICE),
        "seed_modelo_fijo": SEED_MODELO_FIJO,
        "seed_permutacion": SEED_PERMUTACION,
        "n_permutaciones": n_permutaciones,
        "mae_test_real_celsius": mae_real,
        "mae_test_permutado": {
            "media": float(np.mean(mae_permutados)),
            "desviacion_tipica": float(np.std(mae_permutados, ddof=1)) if n_permutaciones > 1 else 0.0,
            "min": float(np.min(mae_permutados)),
            "max": float(np.max(mae_permutados)),
            "valores": mae_permutados.tolist(),
        },
        "n_permutaciones_que_igualan_o_superan_lo_real": n_iguala_o_mejora,
        "p_valor_empirico": p_valor,
    }

    (RESULTS_DIR / "permutation_test_torch_metrics.json").write_text(
        json.dumps(resumen, indent=2, ensure_ascii=False), encoding="utf-8")
    graficar_histograma(mae_permutados, mae_real, p_valor, RESULTS_DIR / "permutation_test_torch.png")

    print(f"\nMAE real: {mae_real:.4f} °C")
    print(f"MAE barajado: media={resumen['mae_test_permutado']['media']:.4f}, "
          f"min={resumen['mae_test_permutado']['min']:.4f}, max={resumen['mae_test_permutado']['max']:.4f}")
    print(f"p-valor empirico: {p_valor:.4g} ({n_iguala_o_mejora}/{n_permutaciones} barajados "
          f"igualan o superan el resultado real)")
    print(f"Resultados guardados en {RESULTS_DIR}")

    return resumen


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=N_PERMUTACIONES_DEFAULT)
    args = parser.parse_args()
    main(n_permutaciones=args.n)
