import numpy,scipy,scipy.linalg

def svd_scipy_exp_eig_A_B(A, B, alpha, t_vec, stop_after_maximum):
    
    # Décomposition en valeur et vecteur propres Av = cBv
    [c,vecp] = scipy.linalg.eig(A,B)
    vecp_inv = numpy.linalg.inv(vecp)
    
    # Vérifie qu'il n'y ait pas de modes instables ni stationnaires
    c_unstable = c[numpy.where(numpy.imag(c)>=0)]
    if len(c_unstable)!=0:
        print("Il y a {:d} modes instables ou stationnaires: ".format(len(c_unstable)))
        print(c_unstable)
    else:
        print("Il n'y a pas de modes instables ni stationnaires")
    
    c = -1j*alpha*c # pour avoir les valeurs propres de M = -i*alpha*B^-1*A
    ss = numpy.zeros((len(t_vec), len(A)))
    
    for i,t in enumerate(t_vec):
        
        # Calcul de exp(M*t)
        D = scipy.eye(len(A)) * numpy.exp(c*t)
        expMt = numpy.dot( numpy.dot(vecp,D) , vecp_inv )
    
        # Svd de exp(M*t)
        s = scipy.linalg.svd(expMt,compute_uv=False)
        ss[i,:] = s
        
        if stop_after_maximum:
            if ss[i,1]<=ss[i-1,1]:
                break
        
    return ss



def svd_scipy_exp_eig(M, t_vec, stop_after_maximum):
    
    # Décomposition en valeur et vecteur propres de M
    [c,vecp] = scipy.linalg.eig(M)
    vecp_inv = numpy.linalg.inv(vecp)
    
    # Vérifie qu'il n'y ait pas de modes instables ni stationnaires
    c_unstable = c[numpy.where(numpy.imag(c)>=0)]
    if len(c_unstable)!=0:
        print("Il y a {:d} modes instables ou stationnaires: ".format(len(c_unstable)))
        print(c_unstable)
    else:
        print("Il n'y a pas de modes instables ni stationnaires")
    
    ss = numpy.zeros((len(t_vec), len(M)))
    
    for i,t in enumerate(t_vec):
        
        # Calcul de exp(M*t)
        D = scipy.eye(len(M)) * numpy.exp(c*t)
        expMt = numpy.dot( numpy.dot(vecp,D) , vecp_inv )
    
        # Svd de exp(M*t)
        u,s,v = scipy.linalg.svd(expMt)
        ss[i,:] = s
        
        if stop_after_maximum:
            if ss[i,1]<=ss[i-1,1]:
                break
        
    return ss