import tensorflow as tf
import os

base = os.path.dirname(__file__)

#keep the ./ in case base is empty
try:
    _lib = tf.load_op_library(os.path.join(base, "./build/libcholesky_update.so"))
except Exception as e:
    _lib = tf.load_op_library(os.path.join(base, "./libcholesky_update.so"))
_chol_update = _lib.chol_update

def cholesky_update(x, mask, init=1e-5):
    """Create a variable `L` that is the cholesky decomposition of a matrix `A = LL^T`

    if `x` has dimensions `[b,dim]` then mask must have dimensions `[b]` and `L` will
    have dimensions `[b,dim,dim]`.
    
    Arguments:
        x {[Tensor]} -- New sample with batch `b` and dimension `dim`
        mask {[Tensor]} -- Vector mask of length `b` indicating
                            if the current element of `x` should be used in the update.
                            That is if `mask[i] == 0`, then `x[i]` is not used to update `L[i]`.
    
    Keyword Arguments:
        init {[float]} -- Init value for diagonal `L` matrix. Must be non zero for well conditioned updates
                            (default: {1e-5})
    
    Returns:
        L {[Variable]} -- The variable being updated. Use `tf.matmul(L,tf.transpose(L))` to recover `A`.
        update_op -- Update operation that updates `L`
    """

    input_shape = x.get_shape().as_list()

    assert len(input_shape) == 2, "rank of x must be 2"

    L_init = tf.eye(input_shape[1],batch_shape=[input_shape[0]])*init

    L = tf.get_variable("L", initializer=L_init)

    update_op = _chol_update(L, x, mask)

    return L, update_op