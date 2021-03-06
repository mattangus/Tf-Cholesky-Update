import tensorflow as tf
import numpy as np
import cholesky_update

n = 200
m = 100
k = 30

#chol = [np.eye(k) for _ in range(m)]
data = np.random.randint(0, 10, (n,m,k))
mask_v = np.ones([n,m])# np.random.choice(a=[False, True], size=(n,m))
mean = np.mean(data,0)

# L = tf.get_variable("L", initializer=np.array(chol, dtype=np.float32))
x = tf.placeholder(tf.float32, shape=[m,k], name="x")
mask = tf.placeholder(tf.bool, shape=[m], name="mask")

L, update_op = cholesky_update.cholesky_update(x, mask)
#print(update_op)

config = tf.ConfigProto(log_device_placement = True)
#config.graph_options.optimizer_options.opt_level = -1

with tf.Session(config=config) as sess:
    sess.run(tf.local_variables_initializer())
    for i in range(n):
        feed = {x: data[i] - mean, mask: mask_v[i]}
        sess.run(update_op, feed_dict=feed)
    chol = sess.run(L)

expected = np.mean([np.cov(data[:,i,:].T) for i in range(m)],0)
result = np.mean([np.matmul(c.T, c) for c in chol],0)/(n-1)

abs_diff = np.abs(expected - result)

print("max:", np.max(abs_diff))
print("mean:", np.mean(abs_diff))