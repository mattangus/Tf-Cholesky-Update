# Tf-Cholesky-Update
The [Cholesky decomposition](https://en.wikipedia.org/wiki/Cholesky_decomposition) is defined as the decompostition of a matrix `A` into the form `A = LL^T`, or equivalently `A = R^TR` where `L = R^T`. Where `L` is lower triangular with real positive diagonal entries (upper triangular for `R`).

This repo implements the [Rank One Update](https://en.wikipedia.org/wiki/Cholesky_decomposition#Rank-one_update) of the Cholesky decompostion. The update rule is for a matrix `A' = A + xx^T`, can be defined in terms of `L` or `R`.

## Main purpose
I implemented this function to have a more efficient approximation of the [sample covariance](https://en.wikipedia.org/wiki/Sample_mean_and_covariance#Sample_covariance). Computing the outer product is in `O(n^2)` where as the rank one update is in `O(n(n+1)/2)` time. Asymptotically there is no differnece, but for large batch sizes it can speed things up quite a bit. I have not done a formal test, but my code does run faster with this in place.

## Limitations
* Currently only support rank 2 input `x` with rank 3 output `L`
* Weight argument is required (just use `tf.ones` for now)

## How to build

Run the following commands after cloning this repo:

```
cd /path/to/repo
mkdir build
cd build
cmake ..
make
```

## Usage

With the repo in the python path, import the update function.

```
from cholesky_update import cholesky_update
import tensorflow as tf
import numpy as np

x = tf.placeholder(tf.float32, [100, 10])

#set mask to 1 to include all samples always
L, update_op = cholesky_update(x, tf.ones([100]))

with tf.Session() as sess:
    for i in range(10):
        sess.run(update_op, {x: np.random.rand(100, 10)})

    L_value = sess.run(L)
```

## Test files
* test.py
  * Test the internal function
* chol_as_cov.py
  * Test the usage of cholesky decomposition for covariance matrix