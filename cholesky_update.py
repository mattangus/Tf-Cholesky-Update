import tensorflow as tf
import os

base = os.path.dirname(__file__)

#keep the ./ in case base is empty
try:
    _lib = tf.load_op_library(os.path.join(base, "./build/libcholesky_update.so"))
except Exception as e:
    _lib = tf.load_op_library(os.path.join(base, "./libcholesky_update.so"))
_chol_update = _lib.chol_update

def cholesky_update(x):

    input_shape = x.get_shape().as_list()

    assert len(input_shape) == 2, "rank of x must be 2"

    R_init = tf.eye(input_shape[1],batch_shape=[input_shape[0]])*float(1e-10)

    R = tf.get_variable("R", initializer=R_init)

    update_op = _chol_update(R, x)

    return R, update_op