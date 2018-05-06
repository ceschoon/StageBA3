import numpy,scipy,scipy.linalg
import operateurs_t

def f(t,L,Re,alpha,N,U):
    
    [A,B] = operateurs_t.buildAB_forward(Re, alpha, N, U, t)
    B_inv = numpy.linalg.inv(B)
    M = -1j*alpha*numpy.dot(B_inv,A)
    
    return numpy.dot(M,L)

def svd_scipy_RK4(Re,alpha,N,t_max,dt,step,U):

    nt = int(t_max/dt)+1
    st = numpy.empty((int(nt/step)+1,N-4))
    t_vec = numpy.empty((int(nt/step)+1,1))

    t = 0
    L = scipy.eye(N-4)
    t_vec[0] = 0
    st[0,:] = 1

    for i in range(1,nt+1):
    
        # La matrice L=exp(Mt) est avancé de dt dans le temps avec RK4
        k1 = dt*f(t,L,Re,alpha,N,U)
        k2 = dt*f(t+dt/2,L+k1/2,Re,alpha,N,U)
        k3 = dt*f(t+dt/2,L+k2/2,Re,alpha,N,U)
        k4 = dt*f(t+dt,L+k3,Re,alpha,N,U)
    
        t = t+dt
        L = L + 1/6*(k1+2*k2+2*k3+k4)
    
        # Calcul de svd
        if i%step==0:
            index = int(i/step)
            s = scipy.linalg.svd(L,compute_uv=False)
            st[index,:] = s
            t_vec[index] = t
            
    return st,t_vec