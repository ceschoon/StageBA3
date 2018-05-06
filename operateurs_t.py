import numpy,scipy

def buildD2_forward(N,dy):
    """ Construit la matrice de l'opérateur D2 discrétisé de type forward (et backward) au second ordre 
    avec les bonnes conditions au bord
   
    Paramètres:
    ----------
    N   : Nombres de points de dicrétisation de la grille originale (avec bords)
    dy  : espacement entre les points de la grille de discrétisation
   
    Renvoie:
    -------
    D2  : matrice sparse de l'opérateur D2
    """
    
    D2 = (scipy.eye(N-4,N-4,-1) - 2*scipy.eye(N-4,N-4,0) + scipy.eye(N-4,N-4,1))
    
    D2[0,0] = -7/4
    D2[-1,-1] = -7/4
    
    return D2/dy**2

def buildD4_forward(N,dy):
    """ Construit la matrice de l'opérateur D4 discrétisé de type forward (et backward) au second ordre 
    avec les bonnes conditions au bord
   
    Paramètres:
    ----------
    N   : Nombres de points de dicrétisation de la grille originale (avec bords)
    dy  : espacement entre les points de la grille de discrétisation
   
    Renvoie:
    -------
    D4  : matrice sparse de l'opérateur D4
    """
    
    D4 = (scipy.eye(N-4,N-4,-2) - 4*scipy.eye(N-4,N-4,-1) + 6*scipy.eye(N-4,N-4,0) - 4*scipy.eye(N-4,N-4,1) + \
          scipy.eye(N-4,N-4,2))
        
    D4[0,0] = 5
    D4[1,0] = -15/4
    D4[-1,-1] = 5
    D4[-2,-1] = -15/4
    
    return D4/dy**4

def buildAB_forward(Re, alpha, N, U, t):
    """ Construit les matrices A et B de l'équation d'Orr-Sommerfeld discrétisée Av = cBv avec une implémentation 
    forward et backward des conditions aux bords. Le profil U est libre de varier dans le temps
   
    Paramètres:
    ----------
    Re      : Nombre de Reynolds 
    alpha   : Nombre d'onde de la perturbation
    N       : Nombres de points de dicrétisation de la grille originale (avec bords)
    U       : Profil de vitesse, fonction vectorisée de (y,t)
    t       : Instant auquel A(t) et B sont évaluées
   
    Renvoie:
    -------
    A   
    B   
    """
    dy = 2.0/(N-1)
    y = scipy.linspace(-1,1,N)
    U0 = U(y[2:-2],t) * scipy.eye(N-4,N-4)
    U2 = -2 * scipy.eye(N-4,N-4)
    
    Malpha  = alpha    * scipy.eye(N-4,N-4)
    Malpha2 = alpha**2 * scipy.eye(N-4,N-4)
    Malpha4 = alpha**4 * scipy.eye(N-4,N-4)
    
    D2 = buildD2_forward(N,dy)
    D4 = buildD4_forward(N,dy)
    
    A = numpy.dot(U0,(D2-Malpha2)) - U2 - 1/(1j*alpha*Re) * \
        ( D4 - 2* numpy.dot(D2,Malpha2) + Malpha4 )
    B = D2 - Malpha2
    
    return [A,B]