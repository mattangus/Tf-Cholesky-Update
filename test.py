import tensorflow as tf
import numpy as np
import time

import cholesky_update

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

n = 100
m = 129*250
k = 19

R_init = tf.eye(k,batch_shape=[m])
data = np.random.randint(0, 10, (n,m,k))
mean = np.mean(data,0)

R = tf.get_variable("R", initializer=R_init)
x = tf.placeholder(tf.float32, shape=[m,k], name="x")

with tf.device("cpu:0"):
    update_op = cholesky_update.chol_update(R,x)
print(update_op)

config = tf.ConfigProto(log_device_placement = True)
config.graph_options.optimizer_options.opt_level = -1

with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    chol = sess.run(R_init)
    #may take a while to compue this:
    #expected = compute_expected(chol, data, mean)
    start = time.time()
    for i in range(n):
        feed = {x: data[i] - mean}
        sess.run(update_op, feed_dict=feed)
    chol = sess.run(R)
    print("ellapsed:", time.time() - start)

abs_diff = np.abs(expected - chol)

print("max:", np.max(abs_diff))
print("mean:", np.mean(abs_diff))