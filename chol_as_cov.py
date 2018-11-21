import tensorflow as tf
import numpy as np

try:
    _tutorial = tf.load_op_library('./build/libcholesky_update.so')
except Exception as e:
    _tutorial = tf.load_op_library('./libcholesky_update.so')
chol_update = _tutorial.chol_update

n = 100
m = 10
k = 19

chol = [np.eye(k)*1e-5 for _ in range(m)]
data = np.random.randint(0, 10, (n,m,k))
mean = np.mean(data,0)

R = tf.placeholder(tf.float32, shape=[m,k,k], name="R")
x = tf.placeholder(tf.float32, shape=[m,k], name="x")

update_op = chol_update(R,x)
#print(update_op)

config = tf.ConfigProto(log_device_placement = True)
#config.graph_options.optimizer_options.opt_level = -1

with tf.Session(config=config) as sess:
    for i in range(n):
        feed = {R: chol, x: data[i] - mean}
        chol = sess.run(update_op, feed_dict=feed)

expected = np.mean([np.cov(data[:,i,:].T) for i in range(m)],0)
result = np.mean([np.matmul(c.T, c) for c in chol],0)/(n-1)

abs_diff = np.abs(expected - result)

print("max:", np.max(abs_diff))
print("mean:", np.mean(abs_diff))