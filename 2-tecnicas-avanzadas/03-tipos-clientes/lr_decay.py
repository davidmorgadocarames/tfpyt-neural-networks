"""Constantes compartidas del decaimiento de learning rate: `lr = lr0 * DECAY_RATE **
(epoch // DECAY_EVERY)`, mismo DECAY_RATE=0.9 que RRNN/*/lr_decay.py.

DECAY_EVERY=100 (no 1): este problema es full-batch y necesita miles de epocas para converger
(ver baseline_tf.py) -- decayendo cada epoca el learning rate se desploma a ~0 mucho antes de
converger (accuracy 41.67% verificado antes de esta correccion, ver
../../1-sgd-vs-adam/sgd_vs_adam_customer_tf.py para el mismo hallazgo). Decayendo cada 100
epocas, el mismo DECAY_RATE llega a la zona de convergencia con un learning rate todavia
razonable.
"""

DECAY_RATE = 0.9
DECAY_EVERY = 100
