import numpy as np
from matplotlib import pyplot as plt

alpha_em=1/137
m_electron=0.511E-3

def f_ab(a,b):
    return( 1/((a-b)**2+1)**3 )

def fancy_integral(a, b):
    # Define terms of the expression
    A = 1 + a**2
    B = (1 + a**2)**3
    C = 5 + 3*a**2
    D = (1 + a**2)**2
    E = -1 + 15*a**2 + 12*a**4
    F = a * (1 + a**2) * (-23 + 38*a**2 + 21*a**4)
    G = 4 * (5 + a**2 - 18*a**4 - 6*a**6)
    H = 3 * a * (3 + a**2) * (-1 + 7*a**2)
    I = 7 - 29*a**2 - 12*a**4
    J = a * (7 + 3*a**2)
    
    term1 = (1 + a**2) * (a * B * C - D * E * b +
                          a * (1 + a**2) * (-23 + 38*a**2 + 21*a**4) * b**2 +
                          4 * (5 + a**2 - 18*a**4 - 6*a**6) * b**3 +
                          3 * a * (3 + a**2) * (-1 + 7*a**2) * b**4 +
                          (7 - 29*a**2 - 12*a**4) * b**5 +
                          a * (7 + 3*a**2) * b**6)
    
    term2 = (3 * (1 + a**2)**6 - 
             a * (1 + a**2)**3 * (3 + 26*a**2 + 15*a**4) * b +
             3 * (1 + a**2)**2 * (-7 + 21*a**2 + 23*a**4 + 11*a**6) * b**2 - 
             3 * a * (1 + a**2) * (7 + 69*a**2 + 45*a**4 + 15*a**6) * b**3 +
             (-11 + 132*a**2 + 9*a**4 * (38 + 5*a**2 * (4 + a**2))) * b**4 -
             3 * a * (11 + 73*a**2 + 41*a**4 + 11*a**6) * b**5 +
             3 * (-1 + 27*a**2 + 17*a**4 + 5*a**6) * b**6 -
             a * (15 + 10*a**2 + 3*a**4) * b**7)
    
    term3 = (3 * (1 + a**2)**6 - 
             a * (1 + a**2)**3 * (3 + 26*a**2 + 15*a**4) * b +
             3 * (1 + a**2)**2 * (-7 + 21*a**2 + 23*a**4 + 11*a**6) * b**2 -
             3 * a * (1 + a**2) * (7 + 69*a**2 + 45*a**4 + 15*a**6) * b**3 +
             (-11 + 132*a**2 + 9*a**4 * (38 + 5*a**2 * (4 + a**2))) * b**4 -
             3 * a * (11 + 73*a**2 + 41*a**4 + 11*a**6) * b**5 +
             3 * (-1 + 27*a**2 + 17*a**4 + 5*a**6) * b**6 -
             a * (15 + 10*a**2 + 3*a**4) * b**7)
    
    term4 = (1 + a**2)**3 + 6 * a * (1 + a**2)**2 * b - \
            3 * (1 + 6*a**2 + 5*a**4) * b**2 + \
            4 * a * (3 + 5*a**2) * b**3 - \
            3 * (1 + 5*a**2) * b**4 + \
            6 * a * b**5 - b**6
    
    # Calculate the expression
    numerator = (term1 + 
                 term2 * np.arctan2(a, 1) + 
                 term3 * np.arctan2(1 + a**2 - a*b, b) + 
                 4 * b * (term4 * (np.log(1 + (a - b)**2) - 2 * np.log(b))))
    
    denominator = 8 * (1 + a**2)**3 * (1 + (a - b)**2)**3 * b
    
    result = numerator / denominator
    
    if b<1000:
        return result
    else:
        # This was obtained using a Series expansion about b->\infty in Mathematica
        return ((3081/280 - (3081*a**2)/40)/b**8 - (709*a)/(35*b**7) - 91/(30*b**6))

def tree_level_sigma(k,mV, Zeff):
    # Calculate components of the expression
    
    m_e=m_electron 
    
    pmin=(2*m_e*k-mV**2)/k/2
    Lambda = Zeff*alpha_em*m_electron
    
    a=m_e/Lambda 
    b=mV**2/(2*k*Lambda)

    return( (4.0*np.pi*alpha_em)*((mV**2+2*m_e**2)* (2/3/Lambda/m_e)*(1/k**2)*f_ab(a,b)) )

def rad_tail_sigma(k,mV, Zeff):
    # Calculate components of the expression
    
    m_e=m_electron 
    
    pmin=(2*m_e*k-mV**2)/k/2
    Lambda = Zeff*alpha_em*m_e
    
    a=m_e/Lambda 
    b=mV**2/(2*k*Lambda)
    
    s=2*k*m_e+m_e**2
    
    beta = 2*alpha_em/np.pi*(np.log(s/m_e**2) - 1.0)

    return( (4.0*np.pi*alpha_em)*((mV**2+2*m_e**2)* (2/3/Lambda/m_e)*(1/k**2)*(beta/2)*fancy_integral(a,b)) )


def sigma_atomic(k,mV,Zeff):
    
    sigma_0 = tree_level_sigma(k,mV,Zeff)
    sigma_1 = rad_tail_sigma  (k,mV,Zeff)
    return(sigma_0 + sigma_1)