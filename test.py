import tensorflow as tf
import numpy as np

def cholupdate(R,x):
    p = np.size(x)
    x = x.T
    for k in range(p):
        r = np.sqrt(R[k,k]**2 + x[k]**2)
        c = r/R[k,k]
        s = x[k]/R[k,k]
        R[k,k] = r
        R[k,k+1:p] = (R[k,k+1:p] + s*x[k+1:p])/c
        x[k+1:p] = c*x[k+1:p] - s*R[k, k+1:p]
    return R

def compute_expected(chol, data, mean):
    chol = np.copy(chol)
    data = np.copy(data)
    for p in range(data.shape[0]):
        mean_sub = data[p] - mean
        for i in range(chol.shape[0]):
            chol[i] = cholupdate(chol[i], mean_sub[i])
    return chol

try:
    _tutorial = tf.load_op_library('./build/libcholesky_update.so')
except Exception as e:
    _tutorial = tf.load_op_library('./libcholesky_update.so')
chol_update = _tutorial.chol_update

n = 100
m = 10
k = 19

chol = [np.eye(k)*1e-5 for _ in range(m)]
data = np.array([[[6, 0]],
 [[6, 3]],
 [[4, 4]]])
data = np.random.randint(0, 10, (n,m,k))
mean = np.mean(data,0)

expected = compute_expected(chol, data, mean)
#print(x_data)

R = tf.placeholder(tf.float32, shape=[m,k,k], name="R")
x = tf.placeholder(tf.float32, shape=[m,k], name="x")
with tf.device("gpu:0"):
    update_op = chol_update(R,x)
print(update_op)

config = tf.ConfigProto(log_device_placement = True)
config.graph_options.optimizer_options.opt_level = -1

with tf.Session(config=config) as sess:
    for i in range(n):
        feed = {R: chol, x: data[i] - mean}
        chol = sess.run(update_op, feed_dict=feed)
    # print("expected:\n", expected)
    # print("result:\n", chol)
    print(np.mean(np.square(chol - expected)))