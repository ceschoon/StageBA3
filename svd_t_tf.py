import numpy,scipy,scipy.linalg
import tensorflow as tf
import operateurs_t_tf

def f(t,l,Re,alpha,N,U):
    
    [a,b] = operateurs_t_tf.buildAB_forward(Re, alpha, N, U, t)
    b = b*(1+0.0j)
    
    b_inv = numpy.linalg.inv(b)
    m = -1j*alpha*numpy.dot(b_inv,a)
    l_new = numpy.dot(m,l)

#    sess = tf.Session()

#    A = tf.placeholder(tf.complex128, shape=(N-4, N-4))
#    B = tf.placeholder(tf.complex128, shape=(N-4, N-4))
#    L = tf.placeholder(tf.complex128, shape=(N-4, N-4))

#    with tf.device('/gpu:0'):
#        B_inv = tf.matrix_inverse(B)
#        M = tf.multiply(-1j*alpha,tf.matmul(B_inv,A))
#        L_new = tf.matmul(L,M)
    
#    l_new = sess.run(L_new, feed_dict={A:a,B:b,L:l})
    
#    sess.close()
    
    return l_new

def svd_tf_RK4(Re,alpha,N,t_max,dt,step,U):

    nt = int(t_max/dt)+1
    st = numpy.empty((int(nt/step)+1,N-4))
    t_vec = numpy.empty((int(nt/step)+1,1))

    t = 0
    l = scipy.eye(N-4)
    t_vec[0] = 0
    st[0,:] = 1

    for i in range(1,nt+1):
        
        # La matrice L=exp(Mt) est avancé de dt dans le temps avec RK4
        k1 = dt*f(t,l,Re,alpha,N,U)
        k2 = dt*f(t+dt/2,l+k1/2,Re,alpha,N,U)
        k3 = dt*f(t+dt/2,l+k2/2,Re,alpha,N,U)
        k4 = dt*f(t+dt,l+k3,Re,alpha,N,U)
    
        t = t+dt
        
        sess = tf.Session()
        
        K1 = tf.placeholder(tf.complex128, shape=(N-4, N-4))
        K2 = tf.placeholder(tf.complex128, shape=(N-4, N-4))
        K3 = tf.placeholder(tf.complex128, shape=(N-4, N-4))
        K4 = tf.placeholder(tf.complex128, shape=(N-4, N-4))
        L  = tf.placeholder(tf.complex128, shape=(N-4, N-4))
        
        with tf.device('/gpu:0'):
            L_new = L + 1/6*(K1+2*K2+2*K3+K4)
        
        l = sess.run(L_new, feed_dict={K1:k1,K2:k2,K3:k3,K4:k4,L:l})
           
        sess.close()
        
#        l = l + 1/6*(k1+2*k2+2*k3+k4)
    
        # Calcul de svd
        if i%step==0:
            index = int(i/step)
            s = scipy.linalg.svd(l,compute_uv=False)
            st[index,:] = s
            t_vec[index] = t
            
    return st,t_vec
