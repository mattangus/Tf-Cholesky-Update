import tensorflow as tf

try:
    _lib = tf.load_op_library('./build/libcholesky_update.so')
except Exception as e:
    _lib = tf.load_op_library('./libcholesky_update.so')
chol_update = _lib.chol_update