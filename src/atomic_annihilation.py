import numpy as np
from .physical_constants import alpha_em, m_electron

## FIX ME  (global variables should be loaded in from physicalconstants.py)
#alpha_em=1/137
#m_electron=0.511E-3

def radiative_tail_expression(s_rescaled,mV_rescaled):
    # Calculate components of the expression
    
    s=s_rescaled
    mV=mV_rescaled 
    
    term1 = 1 / (32 * (1 + mV**4)**2 * s**2 * (1 + mV**4 - 2*mV**2*s + s**2)**3)
    
    # Define arctan and log functions
    arc_tan_mV2 = np.arctan(mV**2)
    arc_tan_mV2_minus_s = np.arctan(mV**2 - s)
    log_1_plus_mV4 = np.log(1 + mV**4)
    log_1_plus_mV4_minus_2mV2s_plus_s2 = np.log(1 + mV**4 - 2*mV**2*s + s**2)
    
    # Polynomial calculations
    poly_part1 = (1 + mV**4)**2 * (3*mV**14 - 15*mV**12*s + 
        3*mV**2 * (1 + s**2)**2 * (1 + 5*s**2) - 9*mV**8 * s * (3 + 5*s**2) + 
        mV**10 * (9 + 33*s**2) - 3*mV**4 * s * (3 + 14*s**2 + 11*s**4) + 
        mV**6 * (9 + 38*s**2 + 45*s**4) - s * (-3 + 21*s**2 + 11*s**4 + 3*s**6))
    
    poly_part2 = s * (3*mV**2 + 12*mV**6 + 18*mV**10 + 12*mV**14 + 3*mV**18 + 3*s - 
        3*mV**4 * s - 27*mV**8 * s - 33*mV**12 * s - 12*mV**16 * s - 
        25*mV**2 * s**2 + 3*mV**6 * s**2 + 49*mV**10 * s**2 + 21*mV**14 * s**2 + 
        20*s**3 + 4*mV**4 * s**3 - 56*mV**8 * s**3 - 24*mV**12 * s**3 - 
        3*mV**2 * s**4 + 42*mV**6 * s**4 + 21*mV**10 * s**4 + 5*s**5 - 
        19*mV**4 * s**5 - 12*mV**8 * s**5 + 5*mV**2 * s**6 + 3*mV**6 * s**6 - 
        8 * (1 + mV**4)**2 * s * log_1_plus_mV4 + 
        8 * (1 + mV**4)**2 * s * log_1_plus_mV4_minus_2mV2s_plus_s2)
    
    
    # Compute final result
    result = term1 * (-(poly_part1 * arc_tan_mV2) + poly_part1 * arc_tan_mV2_minus_s + poly_part2)
    
    # Use asymptotic expansion as mV->infty 
    if mV**2>10*s:
        return(-4*s/mV**14) - 175*s**2/(8*mV**16)
    else:
        return result

def radiative_tail_sigma(k,mV,Zeff):
    if k==0:
        return(0)
    m_e=m_electron
    
    s=2*m_e*k+ 2*m_e**2 
    Lambda = Zeff*alpha_em*m_e
    Beta   = 2*alpha_em/np.pi*(np.log(s/m_e**2)-1)
   
    mV_rescaled = mV/np.sqrt(Lambda*k*2)
    s_rescaled  = s/(Lambda*k*2)
    
    #print(s_rescaled)
    #print(mV_rescaled**2)
    
    # Factor of two accounts for splitting function on each leg 
    
    return((2*Beta)*8/3/k/Lambda*radiative_tail_expression(s_rescaled,mV_rescaled)) 
    

def tree_level_annihilation_on_atoms(k,mV, Zeff):
    # Calculate components of the expression
    
    m_e=m_electron 
    s=2*m_e**2+2*m_e*k
    
    pmin=(s-mV**2)/k/2
    Lambda = Zeff*alpha_em*m_electron

    return( 8/3*Lambda**5/k/(pmin**2+Lambda**2)**3 )


def rad_tail_annihilation_on_atoms(k,mV,Zeff):
    
    sigma_0 = tree_level_annihilation_on_atoms(k,mV,Zeff)
    sigma_1 = radiative_tail_sigma(k,mV,Zeff)
    return(sigma_0 + sigma_1)

