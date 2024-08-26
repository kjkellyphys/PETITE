import numpy as np
from scipy.interpolate import LinearNDInterpolator
try:
    from .physical_constants import *
except:
    from physical_constants import *

def lor_prod(p,v):
    return p[0]*v[0] - p[1]*v[1] - p[2]*v[2] - p[3]*v[3]

def invariant_mass(*args):
    psum = np.zeros(4)
    for p in args:
        psum += np.asarray(p)
    return np.sqrt(lor_prod(psum,psum))

# boost v into the restframe of p
# you can show that this expression is equivalent to a boost in an arbitrary direction 
# with gamma = Ep/mp and 3 velocity p/(mp*gamma).
def boost(p,v):
    rsq = np.sqrt(lor_prod(p,p))
    v0 = lor_prod(p,v)/rsq
    c1 = (v[0] + v0)/(rsq + p[0])
    boosted_v = [v0, v[1] - c1*p[1], v[2] - c1*p[2], v[3] - c1*p[3]]
    
    return np.array(boosted_v)

def fl_kf(x,s):
    """
    Kuraev-Fadin lepton structure function from appendix of https://arxiv.org/abs/1607.03210v2
    More reliable places are Nicrosini and Trentadue in their Eq. 7
    """
    
    beta = (2.*alpha_em/np.pi) * (np.log(s/m_electron**2) - 1.)
    return (beta/16.)*((8. + 3.*beta)*np.power(1. - x,beta/2.-1.) - 4.*(1. + x))

def fl_kf_scaled(x,s):
    """
    Kuraev-Fadin lepton structure function from appendix of https://arxiv.org/abs/1607.03210v2
    this version has been multiplied by (1-x)^(1-beta/2) to eliminate the singularity at x=1. 
    This is meant to be used with transformed integration variables that absorb the singularity into the 
    measure
    """
    beta = (2.*alpha_em/np.pi) * (np.log(s/m_electron**2) - 1.)
    return (beta/16.)*((8. + 3.*beta) - 4.*(1. + x)*np.power(1. - x,1.-beta/2.))

def lepton_luminosity_integrand(s, x, y):
    """
    Integrand of f_ll in 1607.03210v2 for the Kuraev-Fadin structure function
    s - CM energy 
    x - fraction of CM energy in the hard collision
    y - fraction of CM energy carried by one of the leptons
    """
    # I think the original factor of 2 that appears in the appendix of Han et al is wrong
    # additionally their integrand is missing a factor of 1/y!
    if (x > 0. and x <= 1.) and (y > 0. and y <= 1.) and x <= y:
        return fl_kf(y,s)*fl_kf(x/y,s)/y
        #return 2.*fl_kf(x/y,s)
    else:
        return 0.

def transformed_lepton_luminosity_integrand(s,y,u):
    """
    This is the integral of f(x)f(y/x)/x 
    The singularity at x=1 has been transformed away
    by a variable change u=(1-x)^beta/2
    Args:
        s - Mandelstam s of the interaction
        y - mV^2/s 
        u - fraction of momentum carried by one of the particles, transformed as described above
    Returns:
        lepton luminosity integrand to be integrated over u
    """
    
    beta = (2.*alpha_em/np.pi) * (np.log(s/m_electron**2) - 1.)
    x = 1.- np.power(u,2./beta)
    
    # The factor 2/beta comes from the variable transformation
    #print(fl_kf(y/x,s),"\t", fl_kf_scaled(x,s),"\t",(1./x) )
    return fl_kf(y/x,s)*fl_kf_scaled(x,s)*(1./x) * (2./beta)


_lumi_int_fill_val = np.nan
try:
    from .lumi_integral_data import *
except:
    from lumi_integral_data import *
log_lumi_integral_list = np.log10(lumi_integral_list+1e-99)
log_lumi_integral_interp = LinearNDInterpolator(list(zip(log_lumi_integral_list[:,0], log_lumi_integral_list[:,1])),log_lumi_integral_list[:,2], fill_value=_lumi_int_fill_val)

def lumi_integral_interp(s, x):
    ll = log_lumi_integral_interp(np.log10(s), np.log10(x))
    if ll == _lumi_int_fill_val:
        return 0.
    else:
        return np.power(10, log_lumi_integral_interp(np.log10(s), np.log10(x)))

def radiative_return_cross_section(s, mA):
    eps = 1.
    betaf = np.sqrt( 1. - 4.*(m_electron**2) / (mA**2) )
    
    # this factor should be equal to 12pi^2 Gamma(A'->ee)/(mA * s)
    #Changed betaf -> (1.0/betaf) on 26/08/2024, fix relative to original published result.
    prefac = (4.*np.pi**2)*(alpha_em*eps**2)*(1.0/betaf)*(3./2. - betaf**2 / 2.)/s
    
    lumi_factor = lumi_integral_interp(s, mA**2 / s)

    return prefac*lumi_factor*3.89379e+08 # 1/GeV^2 -> pb
    
