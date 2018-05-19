#! /home/cedric/Coding/anaconda3/bin/python

import tensorflow as tf
import time,numpy

A = numpy.random.random([1001,1001])

################################################################################
# Avec numpy

start = time.time()
u,s,v = numpy.linalg.svd(A)
print("Temps d'exécution: {:.3f} secondes".format(time.time()-start))
print(s[:5])

################################################################################
# Avec tensorflow

sess = tf.InteractiveSession()
sess.as_default()

with tf.device('/gpu:0'):
    s,v,u = tf.svd(A)
    #B = tf.matmul(A,A)

start = time.time()
for i in range(1):
    res = sess.run(s)
    #res = sess.run(B)
print("Temps d'exécution: {:.3f} secondes".format(time.time()-start))
print(s[:5])

################################################################################
