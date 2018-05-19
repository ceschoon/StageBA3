import numpy,scipy
import tensorflow as tf
import operateurs_t_gpu

def f(t,l,Re,alpha,N,U):
    
    #sess = tf.Session()
    
    [a,b] = operateurs_t_gpu.buildAB_forward(Re, alpha, N, U, t)
    b = b*(1+0.0j)
    
    A = tf.placeholder(tf.complex128, shape=(N-4, N-4))
    B = tf.placeholder(tf.complex128, shape=(N-4, N-4))
    L = tf.placeholder(tf.complex128, shape=(N-4, N-4))

    with tf.device('/gpu:0'):
        B_inv = tf.matrix_inverse(B)
        M = tf.multiply(-1j*alpha,tf.matmul(B_inv,A))
        L_new = tf.matmul(L,M)
    
    l_new = sess.run(L_new, feed_dict={A:a,B:b,L:l})
    
    sess.close()
    
    return l_new

def svd_tf_RK4(Re,alpha,N,t_max,dt,step,U):
    
    sess = tf.Session()

    nt = int(t_max/dt)+1
    st = numpy.empty((int(nt/step)+1,N-4)) # valeurs singulières
    t_vec = numpy.empty((int(nt/step)+1,1))

    t = 0
    l = scipy.eye(N-4)*(1+0.0j)
    L = tf.placeholder(tf.complex128, shape=(N-4, N-4))
    t_vec[0] = 0
    st[0,:] = 1

    for i in range(1,nt+1):
    
        # La matrice L=exp(Mt) est avancé de dt dans le temps avec RK4
        with tf.device('/gpu:0'):
            k1 = f(t,L,Re,alpha,N,U)  
            k2 = f(t+dt/2,tf.add(L,tf.multiply(k1,1/2*(1+0.0j))),Re,alpha,N,U)  
            k3 = f(t+dt/2,tf.add(L,tf.multiply(k2,1/2*(1+0.0j))),Re,alpha,N,U)  
            k4 = f(t+dt,tf.add(L,k3),Re,alpha,N,U)  
    
            # L = L + dt/6*(k1+2*k2+2*k3+k4)
            temp = tf.add(k1,tf.add(tf.multiply(2+0.0j,k2),tf.add(tf.multiply(2+0.0j,k3),k4)))
            L = tf.add(L,tf.multiply(dt/6*(1+0.0j),temp))
    
        t =t+dt
        
        # Calcul de svd
        if i%step==0:
            with tf.device('/cpu:0'):  # svd sur gpu pas disponible sur ma machine
                s = tf.svd(L,compute_uv=False)
            
            st[index,:] = sess.run(s, feed_dict={L:l})
            
            #index = int(i/step)
            #s_numpy = sess.run(s)
            #s_numpy = scipy.linalg.svd(l, compute_uv=False)
            #st[index,:] = s_numpy
            t_vec[index] = t
    
    sess.close()
            
    return st,t_vec