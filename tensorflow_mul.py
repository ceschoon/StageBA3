#! /home/cedric/Coding/anaconda3/bin/python

import numpy as np
import tensorflow as tf
import time

##################################
#numpy

np.random.seed(1)
n = 8001
x = np.array(np.random.randn(n,n), dtype = np.float32)
a = time.time(); x.dot(x); print ("\n numpy:",time.time() - a, "secondes\n")

#################################
#tensorflow

size = 8001
x = np.array(np.random.randn(size, size), dtype = np.float32)

X = tf.placeholder(tf.float32, shape=(size, size), name=None)
Y = tf.matmul(X, X)

sess = tf.Session()

a = time.time()
sess.run(Y, feed_dict={X: x})
print ("\n tensorflow:",time.time() - a,"secondes\n")
###################################"
