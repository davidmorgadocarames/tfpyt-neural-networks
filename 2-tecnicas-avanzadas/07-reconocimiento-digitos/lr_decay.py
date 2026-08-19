"""Constantes compartidas del decaimiento de learning rate: `lr = lr0 * DECAY_RATE **
(step // DECAY_EVERY)`, mismo DECAY_RATE=0.9 que RRNN/07-reconocimiento-digitos/lr_decay.py.

A diferencia de 03/06/04 (full-batch, DECAY_EVERY en EPOCAS), este problema usa mini-batch --
DECAY_EVERY debe expresarse en PASOS DE OPTIMIZADOR (mini-batches), no epocas. baseline_tf.py
calcula steps_per_epoch = ceil(n_train / BATCH_SIZE) y pasa ese valor como decay_steps a
ExponentialDecay -- con decay_steps=1 (decayendo cada mini-batch) el learning rate se desploma
a ~0 dentro de la primera epoca (verificado: accuracy 75.65% antes de esta correccion).
"""

DECAY_RATE = 0.9
