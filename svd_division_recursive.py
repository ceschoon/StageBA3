import numpy

def svd_division_recursive_A_B(svd_method, A, B, alpha, tmin, tmax, target_prec, num_segments):
    
    cur_tmin = tmin
    cur_tmax = tmax
    cur_prec = cur_tmax-cur_tmin
    
    while cur_prec>target_prec:
        
        t = numpy.linspace(cur_tmin,cur_tmax,num_segments+1)        
        s = svd_method(A, B, alpha, t, True)[:,1] # max singular value
        
        index_max = numpy.argmax(s)
        cur_tmin = t[index_max-1]
        cur_tmax = t[index_max+1]
        cur_prec = cur_tmax-cur_tmin
        
    s_max = s[index_max]
    t_max = t[index_max]
        
    return s_max,t_max,cur_prec