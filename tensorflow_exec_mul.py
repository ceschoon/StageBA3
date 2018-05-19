#! /home/cedric/Coding/anaconda3/bin/python

import tensorflow as tf
import time,numpy

A = numpy.random.random([1001,1001])

################################################################################
# Avec numpy

start = time.time()
B = numpy.dot(A,A)
print("Temps d'exécution: {:.3f} secondes".format(time.time()-start))
print(B[1,:5])

################################################################################
# Avec tensorflow

sess = tf.InteractiveSession()
sess.as_default()

with tf.device('/cpu:0'):
    B = tf.matmul(A,A)

start = time.time()
for i in range(1):
    res = B.eval()
print("Temps d'exécution: {:.3f} secondes".format(time.time()-start))
print(res[1,:5])

################################################################################
# Avec tensorflow

sess = tf.InteractiveSession()
sess.as_default()

with tf.device('/gpu:0'):
    B = tf.matmul(A,A)

start = time.time()
for i in range(1):
    res = B.eval()
print("Temps d'exécution: {:.3f} secondes".format(time.time()-start))
print(res[1,:5])

################################################################################
