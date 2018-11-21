import tensorflow as tf
import os

base = os.path.dirname(__file__)

#keep the ./ in case base is empty
try:
    _lib = tf.load_op_library(os.path.join(base, "./build/libcholesky_update.so"))
except Exception as e:
    _lib = tf.load_op_library(os.path.join(base, "./libcholesky_update.so"))
chol_update = _lib.chol_update