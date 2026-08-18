"""
Esquema B: split de train/val/test FIJO (seed_split=SEED_SPLIT_FIJO siempre) frente al esquema
"Actual" que mide run_seed_sweep.py (split y modelo, cada uno con su propia semilla, ambos
variando de forma independiente). Mismo patron que
numpy-neural-networks-from-scratch/run_seed_sweep_esquemaB.py -- ver su docstring para el
porque (aisla "cuanto cambia el resultado solo por reinicializar pesos/orden de batches, con
exactamente la misma particion de datos siempre").

Reutiliza ejecutar_unidad() de run_seed_sweep.py -- mismo motor, solo cambia que semillas se le
pasan desde fuera. Escribe metrics_seed_sweep_esquemaB.json / seed_sweep_curvas_esquemaB.png en
la misma carpeta que el esquema Actual, sin pisar sus archivos.

Uso:
    python run_seed_sweep_esquemaB.py                         # las 8 unidades, N por defecto
    python run_seed_sweep_esquemaB.py --framework torch --n 20
"""

import argparse
import sys

import numpy as np

from run_seed_sweep import N_SEEDS_DEFAULT, UNIDADES, ejecutar_unidad

SEED_SPLIT_FIJO = 42  # nunca varia en este esquema -- misma particion train/val/test siempre


def semillas_modelo_fijo(n, offset):
    rng = np.random.default_rng(2_000_000 + offset)
    modelos = rng.integers(0, 2**31 - 1, size=n)
    return [(SEED_SPLIT_FIJO, int(m)) for m in modelos.tolist()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=N_SEEDS_DEFAULT)
    parser.add_argument("--solo", type=str, default=None)
    parser.add_argument("--framework", type=str, default=None, choices=["tf", "torch"])
    args = parser.parse_args()

    unidades = UNIDADES
    if args.solo is not None:
        unidades = [u for u in unidades if u[0] == args.solo]
    if args.framework is not None:
        unidades = [u for u in unidades if u[2] == args.framework]
    if not unidades:
        print(f"Ninguna unidad coincide con --solo={args.solo} --framework={args.framework}. "
              f"Disponibles: {[u[0] for u in UNIDADES]}")
        sys.exit(1)

    for offset, (nombre, script, framework) in enumerate(unidades):
        print(f"\n=== Unidad (esquema B): {nombre} (N={args.n}) ===", flush=True)
        ejecutar_unidad(nombre, script, args.n, offset, sufijo="_esquemaB", seeds_fn=semillas_modelo_fijo)


if __name__ == "__main__":
    main()
