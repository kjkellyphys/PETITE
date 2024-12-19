import numpy as np
from scipy import integrate, special, optimize
import random
import math
try:
    from .physical_constants import *
except:
    from physical_constants import *


"""
Below we need to be careful to distinguish between "space" and "plane"/projected angles (https://pdg.lbl.gov/2019/reviews/rpp2018-rev-passage-particles-matter.pdf)
the "space" angle is the usual polar angle in 3D with respect to the direction of motion
there are two "plane" angles that measure deflections into x and y directions; these are identically distributed
either variables can be used to calculate multiple scattering deflections, but these 
random variables have different probability distributions associated with them and also different standard deviations to account 
for the fact that one angle is 3D angle and the others are a pair of "2D" angles 
(in the "space" formulation there is also an azimuthal angle phi which is assumed to be uniformly distributed).
The Bethe-Moliere distributions are all for "space" angles
The PDG "theta0" parameter gives the standard deviation for the "plane" angle
The Lynch and Dahl paper also deals with the projected/"plane" angles

Three different multiple scattering formalisms are implemented below. They can 
be exchanged by manually changing get_scattered_momentum function to use 
different versions of generate_moliere_angle which returns a space angle.
The different possibilities are: 
    1) generate_moliere_angle_simplified_alt - Gaussian approximation to Moliere scattering given in Lynch and Dahl '91 (fast, and allegedly most accurate) 
    2) generate_moliere_angle_simplified - Gaussian approximation to Moliere scattering given in the PDG (fast)
    3) generate_moliere_angle - Bethe theory of Moliere scattering that includes rare large angle scatters (slowest, but most accurate)
These functions call various auxialiary functions, which is why there are multiple implementations of, e.g.,  get_chic_squared etc 
"""

def moliere_f0(x):
    """
    Leading order multiple scattering distribution in x = theta^2 / (chi_c^2 B)
    Eq. 27 in Bethe 1952
    """
    return 2.*np.exp(-x)
def moliere_f1(x):
    """
    Next-to-leading order multiple scattering distribution in x = theta^2 / (chi_c^2 B)
    Eq. 28 in Bethe 1952
    """
    if x <= 100.:
        if x>1E-6:
            return 2.*np.exp(-x)*(x-1.)*(special.expi(x)-math.log(x)) - 2.*(1.-2.*np.exp(-x))
        else:
            ## EDGE CASE , avoids too small values in log(x) and expi(x) which cancel off 
            x=1E-6
            return 2.*np.exp(-x)*(x-1.)*(special.expi(x)-math.log(x)) - 2.*(1.-2.*np.exp(-x))
    else:
        # avoids numerical problems when x gets large, accurate to about 1e-8 or better
        return 2./np.power(x,2) + 8./np.power(x,3) + 36./np.power(x,4)

def moliere_f(x, B):
    """
    Distribution in x = theta^2 / (chi_c^2 B); the constant precator 1/2 comes from 
    converting the theta distribution into an x distribution -- see Eq. 25 in Bethe
    """
    return 0.5 * (moliere_f0(x) + moliere_f1(x)/B)


def moliere_cdf(x, B):
    """
    Cumulative distribution function of the Moliere multiple scattering distribution 
    in the variable x = theta^2/(chi_c^2 B)
    """
    # This is actually already correctly normalized because f1 integrates to 0

    ## EDGE CASE 1
    if x>100:
        # Use analytic form when the function call is analytic anyways
        return  1.0-(1.0/B/x + 2.0/B/x**2 + 6.0/B/x**3)
    ## EDGE CASE 2 
    elif x<0.01:
        return moliere_f(0,B)*x
    else:
        integrand = lambda xp: moliere_f(xp, B)
        return integrate.quad(integrand, 0., x)[0]

def inverse_moliere_cdf(u, B):
    """
    Inverse CDF of the Moliere multiple scattering distribution 
    in the variable x = theta^2/(chi_c^2 B)
    """
    if 1-u < 1/B/100:
        # EDGE CASE just use
        # analytic inverse CDF leading term in expansion
        return(1/(1-u)/B)
    
    elif u<0.01:
        return(u)

    else:
        f= lambda x: moliere_cdf(x, B) - u
        guess = 1.
        
        # bracket the root 
        a = guess/2
        b = guess*2
        it = 0
        
        while f(a)*f(b) > 0:
            if f(b) < 0:
                b *= 2
            if f(a) > 0:
                a *= 0.5
            it += 1

            if it > 20:
                print("Failed to bracket root")

                ## CHECK ME : do we want to break or exit system? 
                #sys.exit(0)
                break

            continue

        return optimize.root_scalar(f, x0=guess, bracket=[a,b], method='ridder').root
     


def generate_moliere_x(B):
    """
    Sample from the Moliere multiple scattering distribution for x = theta^2 / (chic^2 B) 
    using the inverse transform method
    The B parameter is defined via Eqs. 23 and 22 in Bethe, 1953 and encodes the thickness 
    of the target and the target atomic properties.
    It can be evaluated with get_capital_B
    """
    
    u = random.random()
    return inverse_moliere_cdf(u, B)

# 
def get_b(t, beta, A, Z, z):
    """
    Compute Moliere's b parameter which encodes thickness of the target traversed, 
    and the atomic screening model - this particular implementation uses 
    Thomas-Fermi model appropriate for atoms with many electrons. 
    Eq. 22 in Bethe, 1953
    Note that Bethe used Gaussian units for his electromagnetic charge
    t - target thickness is measured in g/cm^2 (a common unit for the radiation length)
    beta - velocity in c=1
    A - atomic weight in g/mol (i.e., PDG conventions)
    Z - charge of target nucleus
    z - charge of beam particle
    """
    alpha_EM = alpha_em # = e^2 in Gaussian units
    alpha = z*Z*alpha_EM/beta 
    N0 = n_avogadro 
    expb = 6700. * t * (Z+1.)*np.power(Z,1./3.)*np.power(z,2.) / (np.power(beta,2) * A * (1. + 3.34*alpha**2))
    little_b=math.log(expb)
    
    return little_b

def get_capital_B(t, beta, A, Z, z):
    """
    Solution to Eq. 23 in Bethe, 1953
    This is the only parameter that goes directly into sampling the Moliere distribution
    """

    acc = 1e-3
    b = get_b(t, beta, A, Z, z)

    B = b + math.log(b)

    it = 0
    B = b + math.log(B)
        
    while np.fabs(B - (b + math.log(B))) > acc:
        it += 1
        
        B = b + math.log(B)
            
    #print("number of iterations = ", it)
    return B

# Different versions of chic implemented below are called by different versions of generate_moliere_angle
def get_chic_squared(t, beta, A, Z, z):
    """
    squared critical angle for Rutherford scattering, eq. 10 in Bethe, 1953
    t - target thickness is measured in g/cm^2 (a common unit for the radiation length)
    beta - velocity in c=1
    A - atomic weight in g/mol (i.e., PDG conventions)
    Z - charge of target nucleus
    z - charge of beam particle
    """
    
    me_in_inv_cm = m_electron/hbarc # 511 keV in 1/cm

    p = me_in_inv_cm * beta / np.sqrt(1. - beta**2)

    return 4.*np.pi*n_avogadro * alpha_em**2 * t * Z * (Z+1.) * z**2 / (A * p**2 * beta**2)

def get_chic_squared_alt(t, beta, A, Z, z):
    """
    Eq. 1 in Lynch & Dahl, 1991
    t - path length in g/cm^2 (t[cm] * rho[g/cm^3] = t[g/cm^2])
    beta - particle velocity
    A - atomic weight in g/mol (i.e., PDG conventions)
    Z - nuclear charge number
    z - particle charge (+/- 1 for electrons/positrons)
    """
    me_in_MeV = m_electron/MeV
    p = me_in_MeV * beta / np.sqrt(1. - beta**2) # momentum has to be in MeV, see below Eq. 2 in Lynch & Dahl, 1991
    return 0.157 * Z * (Z+1)*(t/A)*np.power(z/(p*beta),2.)

def get_chia_squared_alt(beta, A, Z, z):
    """
    Eq. 2 in Lynch & Dahl, 1991
    beta - particle velocity
    A - atomic weight in g/mol (i.e., PDG conventions)
    Z - nuclear charge number
    z - particle charge (+/- 1 for electrons/positrons)
    """
    me_in_MeV = m_electron/MeV 
    p = me_in_MeV * beta / np.sqrt(1. - beta**2) # momentum has to be in MeV, see below Eq. 2 in Lynch & Dahl, 1991
    return 2.007e-5 * np.power(Z,2./3.) * (1. + 3.34*np.power(Z*z*alpha_em/beta,2.))/p**2

def generate_moliere_angle(t, beta, A, Z, z):
    """
    Generate the physical angle in radians by sampling from the Moliere distribution
    Note that Bethe used Gaussian units for his electromagnetic charge
    t - target thickness is measured in g/cm^2 (a common unit for the radiation length)
    beta - velocity in c=1
    A - atomic weight in g/mol (i.e., PDG conventions)
    Z - charge of target nucleus
    z - charge of beam particle
    """

    b = get_b(t, beta, A, Z, z)

    ### The algorithm in Bethe's paper assumes a relatively large value of
    ### B , however for very short path lengths this will not always hold
    ### If b<2 we just use the simplified (core gaussian) sampling. 
    if b<2:
        theta= generate_moliere_angle_simplified_alt(t, beta, A, Z, z)
    else: 
        B = get_capital_B(t, beta, A, Z, z)
        x = generate_moliere_x(B)

        # squared critical angle for Rutherford scattering, eq. 10 in Bethe, 1953
        chic2 = get_chic_squared(t, beta, A, Z, z)
    
        theta = random.choice([-1,1])*np.sqrt(x*chic2*B)
    
    return theta

def generate_moliere_angle_simplified(t_over_X0, beta, z):
    """
    Gaussian approximation from the PDG, with width given by Eq. 27.10 in 
    https://pdg.lbl.gov/2005/reviews/passagerpp.pdf
    t_over_X0 is the target thickness in radiation lengths
    """
    p = m_electron * beta / np.sqrt(1.-beta**2)
    theta0 = (13.6 * MeV) * np.fabs(z) * np.sqrt(t_over_X0)*(1. + 0.038*math.log(t_over_X0)) / (beta * p)
    #return random.gauss(0.,theta0)
    # theta0 is the standard deviation for the plane angle, but we want to generate the space angle
    # we rewrite the space distribution: exp(-theta^2/(2theta_0^2)) d (theta^2/2) -> exp(-x lambda) d x, x=theta^2/2, lambda = 1/theta0^2
    #print("Highland theta0 = ", theta0)
    return random.choice([-1,1])*np.sqrt(2.*random.expovariate(1./(theta0**2)))
    #return random.choice([-1,1])*np.sqrt(random.gauss(0.,theta0)**2 + random.gauss(0.,theta0)**2)

def generate_moliere_angle_simplified_alt(t, beta, A, Z, z):
    """
    Lynch and Dahl, 1991
    Eq. 7 - note that there's a typo, it should be sigma^2! not sigma
    t - target thickness is measured in g/cm^2 (a common unit for the radiation length)
    beta - velocity in c=1
    A - atomic weight in g/mol (i.e., PDG conventions)
    Z - charge of target nucleus
    z - charge of beam particle
    """
    F = 0.98
    chic2 = get_chic_squared_alt(t, beta, A, Z, z)
    chia2 = get_chia_squared_alt(beta, A, Z, z)
    omega = chic2/chia2 
    v = 0.5*omega/(1.-F)
    theta0 = np.sqrt(chic2 * ((1.+v)*math.log(1.+v)/v -1)/(1.+F**2))
    #print("Lynch and Dah theta0 = ", theta0)
    # theta0 is the standard deviation for the plane angle, but we want to generate the space angle
    # in the small angle approximation the space angle is theta = sqrt(thetax^2 + thetay^2)
    return random.choice([-1,1])*np.sqrt(random.gauss(0.,theta0)**2 + random.gauss(0.,theta0)**2)


def get_rotation_matrix(v):
    """
    Find a rotation matrix s.t. R v = |v|(0,0,1)
    """
    
    vx = v[0]; vy = v[1]; vz = v[2]
    
    # Find first rotation to set y component to 0
    if np.fabs(vx) > 0. and np.fabs(vy) > 0.:
        ta = np.fabs(vy/vx)
        a = np.arctan(ta)
        
        if vx > 0. and vy > 0.:
            a = -a
        if vx < 0. and vy > 0.:
            a = -(np.pi - a)
        if vx < 0. and vy < 0.:
            a = -(np.pi + a)
        if vx > 0. and vy < 0.:
            a = -(2.*np.pi - a)
              
        ca = np.cos(a)
        sa = np.sin(a)
    elif np.fabs(vy) > 0.:
        ca = 0.
        sa = 1.
    else:
        ca = 1.
        sa = 0.
        
    Ra = np.array([[ca, -sa, 0.],[sa,ca,0.],[0.,0.,1.]])

    # Find second rotation to set x component to 0
    vxp = vx*ca - vy*sa
    if np.fabs(vz) > 0. and np.fabs(vxp) > 0.:
        tb = np.fabs(vxp/vz)
        b = np.arctan(tb)

        if vz > 0. and vxp > 0.:
            b = -b
        if vz < 0. and vxp > 0.:
            b = -(np.pi - b)
        if vz < 0. and vxp < 0.:
            b = -(np.pi + b)
        if vz > 0. and vxp > 0.:
            b = -(2.*np.pi - b)

        cb = np.cos(b)
        sb = np.sin(b)
    elif vxp > 0.:
        cb = 0.
        sb = -1.
    elif vxp < 0.:
        cb = 0.
        sb = 1.
    else:
        cb = 1.
        sb = 0.
        
    Rb = np.array([[cb, 0., sb],[0., 1., 0.],[-sb, 0., cb]])
    
    return np.matmul(Rb,Ra)
   
def get_scattered_momentum_fast(p4, t, A, Z,  rescale_MCS=1):
    """
    generate a multiple-scattered four-vector from an input four-vector p4 
    after the particle has traversed t [g/cm^2] radiation lengths of material with atomic weight A [g/mol] and 
    atomic number Z
    """
    p3 = p4[1:]
    p3_norm = np.linalg.norm(p3)
    
    if p3_norm > 0:
        beta = np.linalg.norm(p3)/p4[0]
    else:
        # particle at rest
        return p4
    
    # rotation matrix that takes the lab-frame 3 vector to the frame where it points along z only
    Rz_inv = get_rotation_matrix(p3)
    # this takes us back to the lab frame
    Rz = np.transpose(Rz_inv)
    
    p4_new = np.zeros(4)
    p4_new[0] = p4[0]
    
    # generate Moliere angles and make rotation matrices
    Z_part = 1.
    
    # this is slow but more precise, since it includes large angle scatters
    #theta = generate_moliere_angle(t, beta, A, Z, Z_part)
    
    # this is fast, but approximate -- it excludes the rare large angle scatters
    theta = generate_moliere_angle_simplified_alt(t, beta, A, Z, Z_part)*rescale_MCS

    phi = random.uniform(0.,2.*np.pi)
    
    cth = np.cos(theta)
    sth = np.sin(theta)
    cph = np.cos(phi)
    sph = np.sin(phi)
       
    Rtheta = np.array([[1., 0., 0.],[0., cth, -sth],[0., sth, cth]])
    Rphi = np.array([[cph, -sph, 0.],[sph, cph, 0.],[0., 0., 1.]])
    
    p3_new = p3_norm * np.matmul(Rphi,np.matmul(Rtheta,np.array([0.,0.,1.])))
    
    # Transform back to the lab frame
    p3_new = np.matmul(Rz, p3_new)
    
    # Construct the new momentum
    p4_new[1:] = p3_new
    
    return p4_new


def get_scattered_momentum_Bethe(p4, t, A, Z, rescale_MCS=1):
    """
    generate a multiple-scattered four-vector from an input four-vector p4 
    after the particle has traversed t [g/cm^2] radiation lengths of material with atomic weight A [g/mol] and 
    atomic number Z
    """
    p3 = p4[1:]
    p3_norm = np.linalg.norm(p3)
    
    if p3_norm > 0:
        beta = np.linalg.norm(p3)/p4[0]
    else:
        # particle at rest
        return p4
    
    # rotation matrix that takes the lab-frame 3 vector to the frame where it points along z only
    Rz_inv = get_rotation_matrix(p3)
    # this takes us back to the lab frame
    Rz = np.transpose(Rz_inv)
    
    p4_new = np.zeros(4)
    p4_new[0] = p4[0]
    
    # generate Moliere angles and make rotation matrices
    Z_part = 1.
    
    # this is slow but more precise, since it includes large angle scatters
    theta = generate_moliere_angle(t, beta, A, Z, Z_part)*rescale_MCS
    
    # this is fast, but approximate -- it excludes the rare large angle scatters
    #theta = generate_moliere_angle_simplified_alt(t, beta, A, Z, Z_part)

    phi = random.uniform(0.,2.*np.pi)
    
    cth = np.cos(theta)
    sth = np.sin(theta)
    cph = np.cos(phi)
    sph = np.sin(phi)
       
    Rtheta = np.array([[1., 0., 0.],[0., cth, -sth],[0., sth, cth]])
    Rphi = np.array([[cph, -sph, 0.],[sph, cph, 0.],[0., 0., 1.]])
    
    p3_new = p3_norm * np.matmul(Rphi,np.matmul(Rtheta,np.array([0.,0.,1.])))
    
    # Transform back to the lab frame
    p3_new = np.matmul(Rz, p3_new)
    
    # Construct the new momentum
    p4_new[1:] = p3_new
    
    return p4_new
