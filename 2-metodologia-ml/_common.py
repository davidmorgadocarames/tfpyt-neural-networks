"""
Utilidades compartidas por los scripts de 2-metodologia-ml/: cargar_modulo() importa
cargar_datos(), build_model(), generar_datos(), etc. directamente de los scripts ya
verificados en ../1-sgd-vs-adam/ (mismo patron que cargar_modulo() en
../1-sgd-vs-adam/run_seed_sweep.py) en vez de duplicar su logica de datos/arquitectura, y
entrenar_dense_mnist() entrena el clasificador denso de referencia (Adam, sobre la muestra
reducida de MNIST) reutilizado por ensembling.py y calibracion.py.
"""

import importlib.util
import sys
from pathlib import Path

SGD_VS_ADAM_DIR = Path(__file__).parent.parent / "1-sgd-vs-adam"


def cargar_modulo(nombre_script):
    ruta = SGD_VS_ADAM_DIR / nombre_script
    carpeta = str(ruta.parent)
    if carpeta not in sys.path:
        sys.path.insert(0, carpeta)
    spec = importlib.util.spec_from_file_location(nombre_script.replace(".py", ""), ruta)
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


def entrenar_dense_mnist(seed_split, seed_modelo, quiet=True):
    """Entrena un clasificador denso (Adam, lr=0.005) sobre la muestra reducida de MNIST de
    sgd_vs_adam_dense_tf.py y devuelve (model, x_test, y_test)."""
    import tensorflow as tf

    dense_tf = cargar_modulo("sgd_vs_adam_dense_tf.py")
    x_train, y_train, x_val, y_val, x_test, y_test = dense_tf.cargar_datos(seed_split, quiet=quiet)

    tf.keras.utils.set_random_seed(seed_modelo)
    tf.config.experimental.enable_op_determinism()
    model = dense_tf.build_model(tf.keras.optimizers.Adam(learning_rate=0.005))
    model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=dense_tf.EPOCHS_MAX,
        batch_size=len(x_train),
        callbacks=[tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=dense_tf.PATIENCE, restore_best_weights=True)],
        verbose=0,
    )
    return model, x_test, y_test
