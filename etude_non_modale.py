#! /opt/anaconda3/envs/tensorflow/bin/python

import time,numpy
import svd_t,svd_t_tf

def U_statique(y,t):
	return 1-y**2

N = 401
alpha = 0.6
Re = 10000
t_max = 0.0005
dt = 0.0001
step = 1

start =  time.time()
st,t_vec = svd_t_tf.svd_tf_RK4(Re,alpha,N,t_max,dt,step,U_statique)
print("Temps d'exécution avec tensorflow: {:.3f} secondes".format(time.time()-start))
print("Valeurs singulières maximales:")
print(st[:,0])

start =  time.time()
st,t_vec = svd_t.svd_scipy_RK4(Re,alpha,N,t_max,dt,step,U_statique)
print("Temps d'exécution avec numpy: {:.3f} secondes".format(time.time()-start))
print("Valeurs singulières maximales:")
print(st[:,0])
