import numpy as np
from scipy.interpolate import interp1d

import numpy as np
from scipy import integrate, special, optimize
import random

import sys


# Below we need to be careful to distinguish between "space" and "plane"/projected angles (https://pdg.lbl.gov/2019/reviews/rpp2018-rev-passage-particles-matter.pdf)
# the "space" angle is the usual polar angle in 3D with respect to the direction of motion
# there are two "plane" angles that measure deflections into x and y directions; these are identically distributed
# either variables can be used to calculate multiple scattering deflections, but these 
# random variables have different probability distributions associated with them and also different standard deviations to account 
# for the fact that one angle is 3D angle and the others are a pair of "2D" angles 
# (in the "space" formulation there is also an azimuthal angle phi which is assumed to be uniformly distributed).
# The Bethe-moliere distributions are all for "space" angles
# The PDG "theta0" parameter gives the standard deviation for the "plane" angle
# The Lynch and Dahl paper also deals with the projected/"plane" angles
# We will stick to the "space" formulation.

def moliere_f0(x):
    """
    Leading order multiple scattering distribution in x = theta^2 / (chi_c^2 B)
    Eq. 27 in Bethe 1952
    """
    return 2.*np.exp(-x)
def moliere_f1(x):
    """
    Next-to-lLeading order multiple scattering distribution in x = theta^2 / (chi_c^2 B)
    Eq. 28 in Bethe 1952
    """
    if x <= 100.:
        return 2.*np.exp(-x)*(x-1.)*(special.expi(x)-np.log(x)) - 2.*(1.-2.*np.exp(-x))
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
    integrand = lambda xp: moliere_f(xp, B)
    return integrate.quad(integrand, 0., x)[0]

def inverse_moliere_cdf(u, B):
    """
    Inverse CDF of the Moliere multiple scattering distribution 
    in the variable x = theta^2/(chi_c^2 B)
    """
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
            sys.exit(0)
        continue

    # The 700 upper bound just comes from numerical experiments: moliere functions have exponentials that behave poorly when the arguments get too big
    if u > 0.9999:
        return 700.
    
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
    alpha_EM = 1./137. # = e^2 in Gaussian units
    alpha = z*Z*alpha_EM/beta 
    N0 = 6.0221409e+23
    expb = 6700. * t * (Z+1.)*np.power(Z,1./3.)*np.power(z,2.) / (np.power(beta,2) * A * (1. + 3.34*alpha**2))
    return np.log(expb)

def get_capital_B(t, beta, A, Z, z):
    """
    Solution to Eq. 23 in Bethe, 1953
    This is the only parameter that goes directly into sampling the Moliere distribution
    """

    acc = 1e-3
    b = get_b(t, beta, A, Z, z)

    B = b + np.log(b)
    it = 0
    while np.fabs(B - (b + np.log(B))) > acc:
        it += 1
        B = b + np.log(B)
    
    #print("number of iterations = ", it)
    return B

def get_chic_squared(t, beta, A, Z, z):
    """
    squared critical angle for Rutherford scattering, eq. 10 in Bethe, 1953
    t - target thickness is measured in g/cm^2 (a common unit for the radiation length)
    beta - velocity in c=1
    A - atomic weight in g/mol (i.e., PDG conventions)
    Z - charge of target nucleus
    z - charge of beam particle
    """
    alpha_EM = 1./137. # = e^2 in Gaussian units
    N0 = 6.0221409e+23
    
    
    me_in_inv_cm = 2.58961e+10 # 511 keV in 1/cm
    p = me_in_inv_cm * beta / np.sqrt(1. - beta**2)
    return 4.*np.pi*N0 * alpha_EM**2 * t * Z * (Z+1.) * z**2 / (A * p**2 * beta**2)

def get_chic_squared_alt(t, beta, A, Z, z):
    # Eq. 1 in Lynch & Dahl, 1991
    me_in_MeV = 0.511
    #print(beta**2, "\t", 1. - beta**2)
    p = me_in_MeV * beta / np.sqrt(1. - beta**2)
    return 0.157 * Z * (Z+1)*(t/A)*np.power(z/(p*beta),2.)

def get_chia_squared_alt(beta, A, Z, z):
    # Eq. 2 in Lynch & Dahl, 1991
    me_in_MeV = 0.511
    p = me_in_MeV * beta / np.sqrt(1. - beta**2)
    alpha_EM = 1./137
    return 2.007e-5 * np.power(Z,2./3.) * (1. + 3.34*np.power(Z*z*alpha_EM/beta,2.))/p**2

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
    me = 511. * 1e-6
    p = me * beta / np.sqrt(1.-beta**2)
    theta0 = (13.6 * 1e-3) * np.fabs(z) * np.sqrt(t_over_X0)*(1. + 0.038*np.log(t_over_X0)) / (beta * p)
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
    """
    F = 0.98
    chic2 = get_chic_squared_alt(t, beta, A, Z, z)
    chia2 = get_chia_squared_alt(beta, A, Z, z)
    omega = chic2/chia2 
    v = 0.5*omega/(1.-F)
    me = 511. * 1e-6
    theta0 = np.sqrt(chic2 * ((1.+v)*np.log(1.+v)/v -1)/(1.+F**2))
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

    #sa = 0.
    #ca = 1.
    
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
    
    #return Rb
    
    #print "Rb v = ", np.matmul(Rb,v)
    
    return np.matmul(Rb,Ra)
   
def get_scattered_momentum(p4, t, A, Z):
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
    theta = generate_moliere_angle_simplified_alt(t, beta, A, Z, Z_part)

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


me = 0.000511

class Particle:
    def __init__(self, PID, E0, px0, py0, pz0, x0, y0, z0, ID, ParID, ParPID, GenID, GenProcess, Weight):
        self.set_IDs(np.array([PID, ID, ParPID, ParID, GenID, GenProcess, Weight]))

        self.set_p0(np.array([E0, px0, py0, pz0]))
        self.set_r0(np.array([x0, y0, z0]))

        self.set_Ended(False)

        self.set_pf(np.array([E0,px0,py0,pz0]))
        self.set_rf(np.array([x0, y0, z0]))

    def set_IDs(self, value):
        self._IDs = value
    def get_IDs(self):
        return self._IDs

    def set_p0(self, value):
        self._p0 = value
    def get_p0(self):
        return self._p0
    def set_pf(self, value):
        self._pf = value
    def get_pf(self):
        return self._pf

    def set_r0(self, value):
        self._r0 = value
    def get_r0(self):
        return self._r0
    def set_rf(self, value):
        self._rf = value
    def get_rf(self):
        return self._rf

    def set_Ended(self, value):    
        if value != True and value != False:
            raise ValueError("Ended property must be a boolean.")
        self._Ended = value
    def get_Ended(self):
        return self._Ended
        
def eegFourVecs(ep, me, w, ct, ctp, ph):
    epp = ep - w
    p, pp = np.sqrt(ep**2 - me**2), np.sqrt(epp**2 - me**2)

    Em4v = [ep, 0, 0, p] #Four-vector of electron
    al = np.random.uniform(0, 2.0*np.pi)
    cal, sal = np.cos(al), np.sin(al)
    st, stp = np.sqrt(1.0 - ct**2), np.sqrt(1.0 - ctp**2)
    sp, cp = np.sin(ph), np.cos(ph)
    g4v = [w, w*cal*st, w*sal*st, w*ct] #Four-vector of photon

    Ep4v = [epp, pp*(sal*sp*stp + cal*(ctp*st - cp*ct*stp)), pp*(ctp*sal*st - (cp*ct*sal + cal*sp)*stp), pp*(ct*ctp + cp*st*stp)] #Four-vector of positron

    return [Em4v, Ep4v, g4v]

def eeVFourVecs(ep, me, w, MV, ct, ctp, ph):
    epp = ep - w
    p, pp, k = np.sqrt(ep**2 - me**2), np.sqrt(epp**2 - me**2), np.sqrt(w**2 - MV**2)

    Em4v = [ep, 0, 0, p] #Four-vector of electron
    al = np.random.uniform(0, 2.0*np.pi)
    cal, sal = np.cos(al), np.sin(al)
    st, stp = np.sqrt(1.0 - ct**2), np.sqrt(1.0 - ctp**2)
    sp, cp = np.sin(ph), np.cos(ph)
    V4v = [w, k*cal*st, k*sal*st, k*ct] #Four-vector of photon

    Ep4v = [epp, pp*(sal*sp*stp + cal*(ctp*st - cp*ct*stp)), pp*(ctp*sal*st - (cp*ct*sal + cal*sp)*stp), pp*(ct*ctp + cp*st*stp)] #Four-vector of positron

    return [Em4v, Ep4v, V4v]

def gepemFourVecs(w, me, epp, ctp, ctm, ph):
    epm = w - epp
    pm, pp = np.sqrt(epm**2 - me**2), np.sqrt(epp**2 - me**2)

    Eg4v = [w, 0, 0, w]
    al = np.random.uniform(0, 2.0*np.pi)

    cal, sal = np.cos(al), np.sin(al)
    stp, stm = np.sqrt(1.0 - ctp**2), np.sqrt(1.0 - ctm**2)
    spal, cpal = np.sin(ph+al), np.cos(ph+al)

    pp4v = [epp, pp*stp*cal, pp*stp*sal, pp*ctp]
    pm4v = [epm, pm*stm*cpal, pm*stm*spal, pm*ctm]

    return [Eg4v, pp4v, pm4v]
    
def Compton_FVs(Eg, me, mV, ct):
    s = me**2 + 2*Eg*me
    Ee0 = (s + me**2)/(2.0*np.sqrt(s))
    Ee = (s - mV**2 + me**2)/(2*np.sqrt(s))
    EV = (s + mV**2 - me**2)/(2*np.sqrt(s))
    pF = np.sqrt(Ee**2 - me**2)

    g0 = Ee0/me
    b0 = 1.0/g0*np.sqrt(g0**2 - 1.0)

    ph = np.random.uniform(0, 2.0*np.pi)
    pe4v = [g0*Ee - b0*g0*pF*ct, -pF*np.sqrt(1-ct**2)*np.sin(ph), -pF*np.sqrt(1-ct**2)*np.cos(ph), b0*g0*Ee-g0*pF*ct]
    pV4v = [g0*EV + b0*g0*pF*ct, pF*np.sqrt(1-ct**2)*np.sin(ph), pF*np.sqrt(1-ct**2)*np.cos(ph), b0*g0*EV + g0*pF*ct]

    return [pe4v, pV4v]

def Ann_FVs(Ee, me, mV, ct):
    s = 2*me*(Ee+me)
    EeCM = np.sqrt(s)/2.0
    Eg = (s - mV**2)/(2*np.sqrt(s))
    EV = (s + mV**2)/(2*np.sqrt(s))
    pF = Eg

    g0 = EeCM/me
    b0 = 1.0/g0*np.sqrt(g0**2-1.0)

    ph = np.random.uniform(0.0, 2.0*np.pi)

    if ct < -1.0 or ct > 1.0:
        print("Error in Annihiliation Calculation")
        print(Ee, me, mV, ct)

    pg4v = [g0*Eg - b0*g0*pF*ct, -pF*np.sqrt(1-ct**2)*np.sin(ph), -pF*np.sqrt(1-ct**2)*np.cos(ph), b0*g0*Eg - g0*pF*ct]
    pV4v = [g0*EV + b0*g0*pF*ct, pF*np.sqrt(1-ct**2)*np.sin(ph), pF*np.sqrt(1-ct**2)*np.cos(ph), b0*g0*EV + g0*pF*ct]

    return [pg4v, pV4v]

Z = {'graphite':6.0, 'lead':82.0} #atomic number of different targets
A = {'graphite':12.0, 'lead':207.2} #atomic mass of different targets
rho = {'graphite':2.210, 'lead':11.35} #g/cm^3
dEdx = {'graphite':2.0*rho['graphite'], 'lead':2.0*rho['lead']} #MeV per cm

GeVsqcm2 = 1.0/(5.06e13)**2 #Conversion between cross sections in GeV^{-2} to cm^2
cmtom = 0.01
mp0 = 1.673e-24 #g

class Shower:
    def __init__(self, PickDir, TargetMaterial, MinEnergy):
        self.set_PickDir(PickDir)
        self.set_TargetMaterial(TargetMaterial)
        self.set_SampDir(PickDir + TargetMaterial + "/")
        self.set_SampDirE(PickDir + "electrons/")
        self.MinEnergy = MinEnergy

        self.set_MaterialProps()
        self.set_nTargets()
        self.set_samples()
        self.set_CrossSections()
        self.set_NSigmas()

    def set_PickDir(self, value):
        self._PickDir = value
    def get_PickDir(self):
        return self._PickDir
    def set_TargetMaterial(self, value):
        self._TargetMaterial = value
    def get_TargetMaterial(self):
        return self._TargetMaterial
    def set_SampDir(self, value):
        self._SampDir = value
    def get_SampDir(self):
        return self._SampDir
    def set_SampDirE(self, value):
        self._SampDirE = value
    def get_SampDirE(self):
        return self._SampDirE

    def set_MaterialProps(self):
        self._ZTarget, self._ATarget, self._rhoTarget, self._dEdx = Z[self.get_TargetMaterial()], A[self.get_TargetMaterial()], rho[self.get_TargetMaterial()], dEdx[self.get_TargetMaterial()]
    def get_MaterialProps(self):
        return self._ZTarget, self._ATarget, self._rhoTarget, self._dEdx

    def set_nTargets(self):
        ZT, AT, rhoT, dEdxT = self.get_MaterialProps()
        self._nTarget = rhoT/mp0/AT
        self._nElecs = self._nTarget*ZT
    def get_nTargets(self):
        return self._nTarget, self._nElecs

    def set_samples(self):
        self._BremSamples = np.load(self.get_SampDir()+"BremEvts.npy", allow_pickle=True)
        self._PPSamples = np.load(self.get_SampDir()+"PairProdEvts.npy", allow_pickle=True)
        self._AnnSamples = np.load(self.get_SampDirE()+"AnnihilationEvts.npy", allow_pickle=True)
        self._CompSamples = np.load(self.get_SampDirE()+"ComptonEvts.npy", allow_pickle=True)
    def get_BremSamples(self, ind):
        return self._BremSamples[ind]
    def get_PPSamples(self, ind):
        return self._PPSamples[ind]
    def get_AnnSamples(self, ind):
        return self._AnnSamples[ind]
    def get_CompSamples(self, ind):
        return self._CompSamples[ind]

    def set_CrossSections(self):
        self._BremXSec = np.load(self.get_SampDir()+"BremXSec.npy", allow_pickle=True)
        self._PPXSec = np.load(self.get_SampDir()+"PairProdXSec.npy", allow_pickle=True)
        self._AnnXSec = np.load(self.get_SampDirE()+"AnnihilationXSec.npy", allow_pickle=True)
        self._CompXSec = np.load(self.get_SampDirE()+"ComptonXSec.npy", allow_pickle=True)

        self._EeVecBrem = np.transpose(self._BremXSec)[0]
        self._EgVecPP = np.transpose(self._PPXSec)[0]
        self._EeVecAnn = np.transpose(self._AnnXSec)[0]
        self._EgVecComp = np.transpose(self._CompXSec)[0]

        self._logEeMinBrem, self._logEeSSBrem = np.log10(self._EeVecBrem[0]), np.log10(self._EeVecBrem[1]) - np.log10(self._EeVecBrem[0])
        self._logEeMinAnn, self._logEeSSAnn = np.log10(self._EeVecAnn[0]), np.log10(self._EeVecAnn[1]) - np.log10(self._EeVecAnn[0])
        self._logEgMinPP, self._logEgSSPP = np.log10(self._EgVecPP[0]), np.log10(self._EgVecPP[1]) - np.log10(self._EgVecPP[0])
        self._logEgMinComp, self._logEgSSComp= np.log10(self._EgVecComp[0]), np.log10(self._EgVecComp[1]) - np.log10(self._EgVecComp[0])

    def get_BremXSec(self):
        return self._BremXSec
    def get_PPXSec(self):
        return self._PPXSec
    def get_AnnXSec(self):
        return self._AnnXSec
    def get_CompXSec(self):
        return self._CompXSec

    def set_NSigmas(self):
        BS, PPS, AnnS, CS = self.get_BremXSec(), self.get_PPXSec(), self.get_AnnXSec(), self.get_CompXSec()
        nZ, ne = self.get_nTargets()
        self._NSigmaBrem = interp1d(np.transpose(BS)[0], nZ*GeVsqcm2*np.transpose(BS)[1])
        self._NSigmaPP = interp1d(np.transpose(PPS)[0], nZ*GeVsqcm2*np.transpose(PPS)[1])
        self._NSigmaAnn = interp1d(np.transpose(AnnS)[0], ne*GeVsqcm2*np.transpose(AnnS)[1])
        self._NSigmaComp = interp1d(np.transpose(CS)[0], ne*GeVsqcm2*np.transpose(CS)[1])

    def GetMFP(self, PID, Energy):
        if PID == 22:
            return cmtom*(self._NSigmaPP(Energy) + self._NSigmaComp(Energy))**-1
        elif PID == 11:
            return cmtom*(self._NSigmaBrem(Energy))**-1
        elif PID == -11:
            return cmtom*(self._NSigmaBrem(Energy) + self._NSigmaAnn(Energy))**-1
        
    def BF_Positron_Brem(self, Energy):
        b0, b1 = self._NSigmaBrem(Energy), self._NSigmaAnn(Energy)
        return b0/(b0+b1)
    def BF_Photon_PP(self, Energy):
        b0, b1 = self._NSigmaPP(Energy), self._NSigmaComp(Energy)
        return b0/(b0+b1)

    def ElecBremSample(self, Elec0):
        Ee0, pex0, pey0, pez0 = Elec0.get_pf()

        ThZ = np.arccos(pez0/np.sqrt(pex0**2 + pey0**2 + pez0**2))
        PhiZ = np.arctan2(pey0, pex0)
        RM = [[np.cos(ThZ)*np.cos(PhiZ), -np.sin(PhiZ), np.sin(ThZ)*np.cos(PhiZ)],
            [np.cos(ThZ)*np.sin(PhiZ), np.cos(PhiZ), np.sin(ThZ)*np.sin(PhiZ)],
            [-np.sin(ThZ), 0, np.cos(ThZ)]]

        LUKey = int((np.log10(Ee0) - self._logEeMinBrem)/self._logEeSSBrem)
        ts = self.get_BremSamples(LUKey)
        SampEvt = ts[np.random.randint(0, len(ts))]
        EeMod = self._EeVecBrem[LUKey]

        NFVs = eegFourVecs(Ee0, me, SampEvt[0]*Ee0/EeMod, np.cos(me/EeMod*SampEvt[1]), np.cos(me/(Ee0-SampEvt[0]*Ee0/EeMod)*SampEvt[2]), SampEvt[3])

        Eef, pexfZF, peyfZF, pezfZF = NFVs[1]
        Egf, pgxfZF, pgyfZF, pgzfZF = NFVs[2]

        pe3ZF = [pexfZF, peyfZF, pezfZF]
        pg3ZF = [pgxfZF, pgyfZF, pgzfZF]
        
        pe3LF = np.dot(RM, pe3ZF)
        pg3LF = np.dot(RM, pg3ZF)
        
        NewE = Particle(Elec0.get_IDs()[0], Eef, pe3LF[0], pe3LF[1], pe3LF[2], Elec0.get_rf()[0], Elec0.get_rf()[1], Elec0.get_rf()[2], 2*(Elec0.get_IDs()[1])+0, Elec0.get_IDs()[1], Elec0.get_IDs()[0], Elec0.get_IDs()[4]+1, 0, 1.0)
        NewG = Particle(22, Egf, pg3LF[0], pg3LF[1], pg3LF[2], Elec0.get_rf()[0], Elec0.get_rf()[1], Elec0.get_rf()[2], 2*(Elec0.get_IDs()[1])+1, Elec0.get_IDs()[1], Elec0.get_IDs()[0], Elec0.get_IDs()[4]+1, 0, 1.0)

        return [NewE, NewG]

    def AnnihilationSample(self, Elec0):
        Ee0, pex0, pey0, pez0 = Elec0.get_pf()

        ThZ = np.arccos(pez0/np.sqrt(pex0**2 + pey0**2 + pez0**2))
        PhiZ = np.arctan2(pey0, pex0)
        RM = [[np.cos(ThZ)*np.cos(PhiZ), -np.sin(PhiZ), np.sin(ThZ)*np.cos(PhiZ)],
            [np.cos(ThZ)*np.sin(PhiZ), np.cos(PhiZ), np.sin(ThZ)*np.sin(PhiZ)],
            [-np.sin(ThZ), 0, np.cos(ThZ)]]

        LUKey = int((np.log10(Ee0) - self._logEeMinAnn)/self._logEeSSAnn)
        ts = self.get_AnnSamples(LUKey)
        SampEvt = ts[np.random.randint(0, len(ts))]
        EeMod = self._EeVecAnn[LUKey]

        NFVs = Ann_FVs(Ee0, me, 0.0, SampEvt[0])

        Eg1f, pg1xfZF, pg1yfZF, pg1zfZF = NFVs[0]
        Eg2f, pg2xfZF, pg2yfZF, pg2zfZF = NFVs[1]

        pg3ZF1 = [pg1xfZF, pg1yfZF, pg1zfZF]
        pg3ZF2 = [pg2xfZF, pg2yfZF, pg2zfZF]
    
        pg3LF1 = np.dot(RM, pg3ZF1)
        pg3LF2 = np.dot(RM, pg3ZF2)   

        NewG1 = Particle(22, Eg1f, pg3LF1[0], pg3LF1[1], pg3LF1[2], Elec0.get_rf()[0], Elec0.get_rf()[1], Elec0.get_rf()[2], 2*(Elec0.get_IDs()[1])+0, Elec0.get_IDs()[1], Elec0.get_IDs()[0], Elec0.get_IDs()[4]+1, 1, 1.0)
        NewG2 = Particle(22, Eg2f, pg3LF2[0], pg3LF2[1], pg3LF2[2], Elec0.get_rf()[0], Elec0.get_rf()[1], Elec0.get_rf()[2], 2*(Elec0.get_IDs()[1])+1, Elec0.get_IDs()[1], Elec0.get_IDs()[0], Elec0.get_IDs()[4]+1, 1, 1.0)

        return [NewG1, NewG2]

    def PhotonSplitSample(self, Phot0):
        Eg0, pgx0, pgy0, pgz0 = Phot0.get_pf()

        ThZ = np.arccos(pgz0/np.sqrt(pgx0**2 + pgy0**2 + pgz0**2))
        PhiZ = np.arctan2(pgy0, pgx0)
        RM = [[np.cos(ThZ)*np.cos(PhiZ), -np.sin(PhiZ), np.sin(ThZ)*np.cos(PhiZ)],
            [np.cos(ThZ)*np.sin(PhiZ), np.cos(PhiZ), np.sin(ThZ)*np.sin(PhiZ)],
            [-np.sin(ThZ), 0, np.cos(ThZ)]]

        LUKey = int((np.log10(Eg0) - self._logEgMinPP)/self._logEgSSPP)
        ts = self.get_PPSamples(LUKey)
        SampEvt = ts[np.random.randint(0, len(ts))]
        EgMod = self._EgVecPP[LUKey]

        NFVs = gepemFourVecs(Eg0, me, SampEvt[0]*Eg0/EgMod, np.cos(me/EgMod*SampEvt[1]), np.cos(me/EgMod*SampEvt[2]), SampEvt[3])
        Eepf, pepxfZF, pepyfZF, pepzfZF = NFVs[1]
        Eemf, pemxfZF, pemyfZF, pemzfZF = NFVs[2]

        pep3ZF = [pepxfZF, pepyfZF, pepzfZF]
        pem3ZF = [pemxfZF, pemyfZF, pemzfZF]

        pep3LF = np.dot(RM, pep3ZF)
        pem3LF = np.dot(RM, pem3ZF)

        NewEp = Particle(-11,Eepf, pep3LF[0], pep3LF[1], pep3LF[2], Phot0.get_rf()[0], Phot0.get_rf()[1], Phot0.get_rf()[2], 2*(Phot0.get_IDs()[1])+0, Phot0.get_IDs()[1], Phot0.get_IDs()[0], Phot0.get_IDs()[4]+1, 2, 1.0)
        NewEm = Particle(11, Eemf, pem3LF[0], pem3LF[1], pem3LF[2], Phot0.get_rf()[0], Phot0.get_rf()[1], Phot0.get_rf()[2], 2*(Phot0.get_IDs()[1])+1, Phot0.get_IDs()[1], Phot0.get_IDs()[0], Phot0.get_IDs()[4]+1, 2, 1.0)

        return [NewEp, NewEm]

    def ComptonSample(self, Phot0):
        Eg0, pgx0, pgy0, pgz0 = Phot0.get_pf()

        ThZ = np.arccos(pgz0/np.sqrt(pgx0**2 + pgy0**2 + pgz0**2))
        PhiZ = np.arctan2(pgy0, pgx0)
        RM = [[np.cos(ThZ)*np.cos(PhiZ), -np.sin(PhiZ), np.sin(ThZ)*np.cos(PhiZ)],
            [np.cos(ThZ)*np.sin(PhiZ), np.cos(PhiZ), np.sin(ThZ)*np.sin(PhiZ)],
            [-np.sin(ThZ), 0, np.cos(ThZ)]]

        LUKey = int((np.log10(Eg0) - self._logEgMinComp)/self._logEgSSComp)
        ts = self.get_CompSamples(LUKey)
        SampEvt = ts[np.random.randint(0, len(ts))]
        EgMod = self._EgVecComp[LUKey]

        NFVs = Compton_FVs(Eg0, me, 0.0, SampEvt[0])

        Eef, pexfZF, peyfZF, pezfZF = NFVs[0]
        Egf, pgxfZF, pgyfZF, pgzfZF = NFVs[1]


        pe3LF = np.dot(RM, [pexfZF, peyfZF, pezfZF])
        pg3LF = np.dot(RM, [pgxfZF, pgyfZF, pgzfZF])

        NewE = Particle(11, Eef, pe3LF[0], pe3LF[1], pe3LF[2], Phot0.get_rf()[0], Phot0.get_rf()[1], Phot0.get_rf()[2], 2*(Phot0.get_IDs()[1])+0, Phot0.get_IDs()[1], Phot0.get_IDs()[0], Phot0.get_IDs()[4]+1, 3, 1.0)
        NewG = Particle(22, Egf, pg3LF[0], pg3LF[1], pg3LF[2], Phot0.get_rf()[0], Phot0.get_rf()[1], Phot0.get_rf()[2], 2*(Phot0.get_IDs()[1])+1, Phot0.get_IDs()[1], Phot0.get_IDs()[0], Phot0.get_IDs()[4]+1, 3, 1.0)

        return [NewE, NewG]

    def PropagateParticle(self, Part0, Losses=False, MS=False):
        if Part0.get_Ended() is True:
            Part0.set_rf(Part0.get_rf())
            return Part0
        else:
            mfp = self.GetMFP(Part0.get_IDs()[0], Part0.get_p0()[0])
            distC = np.random.uniform(0.0, 1.0)
            dist = mfp*np.log(1.0/(1.0-distC))
            if np.abs(Part0.get_IDs()[0]) == 11:
                M0 = me
            elif Part0.get_IDs()[0] == 22:
                M0 = 0.0

            E0, px0, py0, pz0 = Part0.get_p0()
            if MS:
                ZT, AT, rhoT, dEdxT = self.get_MaterialProps()
                EF0, PxF0, PyF0, PzF0 = get_scattered_momentum(Part0.get_p0(), rhoT*(dist/cmtom), AT, ZT)
                PHatDenom = np.sqrt((PxF0 + px0)**2 + (PyF0 + py0)**2 + (PzF0 + pz0)**2)
                PHat = [(PxF0 + px0)/PHatDenom, (PyF0 + py0)/PHatDenom, (PzF0 + pz0)/PHatDenom]
            else:
                PHatDenom = np.sqrt(px0**2 + py0**2 + pz0**2)
                PHat = [(px0)/PHatDenom, (py0)/PHatDenom, (pz0)/PHatDenom]

            p30 = np.sqrt(px0**2 + py0**2 + pz0**2)

            x0, y0, z0 = Part0.get_r0()
            Part0.set_rf([x0 + PHat[0]*dist, y0 + PHat[1]*dist, z0 + PHat[2]*dist])

            if Losses is False:
                if MS:
                    Part0.set_pf(np.array([E0, PxF0, PyF0, PzF0]))
                else:
                    Part0.set_pf(Part0.get_p0())
            else:
                Ef = E0 - Losses*dist
                if Ef <= M0 or Ef < self.MinEnergy:
                    #print("Particle lost too much energy along path of propagation!")
                    Part0.set_Ended(True)
                    return Part0
                Part0.set_pf(np.array([Ef, px0/p30*np.sqrt(Ef**2-M0**2), py0/p30*np.sqrt(Ef**2-M0**2), pz0/p30*np.sqrt(Ef**2-M0**2)]))

            Part0.set_Ended(True)
            return Part0

    def GenShower(self, PID0, p40, ParPID, VB=False):
        p0 = Particle(PID0, p40[0], p40[1], p40[2], p40[3], 0.0, 0.0, 0.0, 1, 0, ParPID, 0, -1, 1.0)
        if VB:
            print("Starting shower, initial particle with ID Info")
            print(p0.get_IDs())
            print("Initial four-momenta:")
            print(p0.get_p0())

        AllParticles = [p0]

        if p0.get_p0()[0] < self.MinEnergy:
            p0.set_Ended(True)
            return AllParticles

        while all([ap.get_Ended() == True for ap in AllParticles]) is False:
            for apI, ap in enumerate(AllParticles):
                if ap.get_Ended() is True:
                    continue
                else:
                    if ap.get_IDs()[0] == 22:
                        ap = self.PropagateParticle(ap)
                    elif np.abs(ap.get_IDs()[0]) == 11:
                        dEdxT = self.get_MaterialProps()[3]*(0.1) #Converting MeV/cm to GeV/m
                        ap = self.PropagateParticle(ap, MS=True, Losses=dEdxT)
                    AllParticles[apI] = ap
                    if (all([ap.get_Ended() == True for ap in AllParticles]) is True and ap.get_pf()[0] < self.MinEnergy):
                        break
                    if ap.get_IDs()[0] == 11:
                        npart = self.ElecBremSample(ap)
                    elif ap.get_IDs()[0] == -11:
                        BFEpBrem = self.BF_Positron_Brem(ap.get_pf()[0])
                        ch = np.random.uniform(low=0., high=1.0)
                        if ch < BFEpBrem:
                            npart = self.ElecBremSample(ap)
                        else:
                            npart = self.AnnihilationSample(ap)
                    elif ap.get_IDs()[0] == 22:
                        BFPhPP = self.BF_Photon_PP(ap.get_pf()[0])
                        ch = np.random.uniform(low=0., high=1.)
                        if ch < BFPhPP:
                            npart = self.PhotonSplitSample(ap)
                        else:
                            npart = self.ComptonSample(ap)
                    if (npart[0]).get_p0()[0] > self.MinEnergy:
                        AllParticles.append(npart[0])
                    if (npart[1]).get_p0()[0] > self.MinEnergy:
                        AllParticles.append(npart[1])

        return AllParticles

MVLib = {'10MeV':0.010,'100MeV':0.100}
class DarkShower(Shower):
    def __init__(self, PickDir, TargetMaterial, MinEnergy, MVStr):
        super().__init__(PickDir, TargetMaterial, MinEnergy)

        self.set_MV(MVStr)
        self.set_DarkPickDir(PickDir)
        self.set_DarkSampDir(PickDir + TargetMaterial + "/DarkV/")
        self.set_DarkSampDirE(PickDir + "electrons/DarkV/")

        self.set_Darksamples()
        self.set_DarkCrossSections()
        self.set_DarkNSigmas()

    def set_DarkPickDir(self, value):
        self._DarkPickDir = value
    def get_DarkPickDir(self):
        return self._DarkPickDir
    def set_DarkSampDir(self, value):
        self._DarkSampDir = value
    def get_DarkSampDir(self):
        return self._DarkSampDir
    def set_DarkSampDirE(self, value):
        self._DarkSampDirE = value
    def get_DarkSampDirE(self):
        return self._DarkSampDirE
    def set_MV(self, value):
        self._MVStr = value
        self._MV = MVLib[value]
    def get_MVStr(self):
        return self._MVStr
    def get_MV(self):
        return self._MV

    def set_Darksamples(self):
        self._DarkBremSamples = np.load(self.get_DarkSampDir()+"DarkBremEvts_" + self.get_MVStr() + ".npy", allow_pickle=True)
        self._DarkAnnSamples = np.load(self.get_DarkSampDirE()+"AnnihilationEvts_" + self.get_MVStr() + ".npy", allow_pickle=True)
        self._DarkCompSamples = np.load(self.get_DarkSampDirE()+"ComptonEvts_" + self.get_MVStr() + ".npy", allow_pickle=True)
    def get_DarkBremSamples(self, ind):
        return self._DarkBremSamples[ind]
    def get_DarkAnnSamples(self, ind):
        return self._DarkAnnSamples[ind]
    def get_DarkCompSamples(self, ind):
        return self._DarkCompSamples[ind]
    
    def set_DarkCrossSections(self):
        self._DarkBremXSec = np.load(self.get_DarkSampDir()+"DarkBremXSec_"+self.get_MVStr()+".npy",allow_pickle=True)
        self._DarkAnnXSec = np.load(self.get_DarkSampDirE()+"AnnihilationXSec_"+self.get_MVStr()+".npy",allow_pickle=True)
        self._DarkCompXSec = np.load(self.get_DarkSampDirE()+"ComptonXSec_"+self.get_MVStr()+".npy",allow_pickle=True)

        self._EeVecDarkBrem = np.transpose(self._DarkBremXSec)[0]
        self._EeVecDarkAnn = np.transpose(self._DarkAnnXSec)[0]
        self._EgVecDarkComp = np.transpose(self._DarkCompXSec)[0]

        self._logEeMinDarkBrem, self._logEeSSDarkBrem = np.log10(self._EeVecDarkBrem[0]), np.log10(self._EeVecDarkBrem[1]) - np.log10(self._EeVecDarkBrem[0])
        self._logEeMinDarkAnn, self._logEeSSDarkAnn = np.log10(self._EeVecDarkAnn[0]), np.log10(self._EeVecDarkAnn[1]) - np.log10(self._EeVecDarkAnn[0])
        self._logEgMinDarkComp, self._logEgSSDarkComp = np.log10(self._EgVecDarkComp[0]), np.log10(self._EgVecDarkComp[1]) - np.log10(self._EgVecDarkComp[0])

    def get_DarkBremXSec(self):
        return self._DarkBremXSec
    def get_DarkAnnXSec(self):
        return self._DarkAnnXSec
    def get_DarkCompXSec(self):
        return self._DarkCompXSec

    def set_DarkNSigmas(self):
        DBS, DAnnS, DCS = self.get_DarkBremXSec(), self.get_DarkAnnXSec(), self.get_DarkCompXSec()
        nZ, ne = self.get_nTargets()
        self._NSigmaDarkBrem = interp1d(np.transpose(DBS)[0], nZ*GeVsqcm2*np.transpose(DBS)[1])
        self._NSigmaDarkAnn = interp1d(np.transpose(DAnnS)[0], ne*GeVsqcm2*np.transpose(DAnnS)[1])
        self._NSigmaDarkComp = interp1d(np.transpose(DCS)[0], ne*GeVsqcm2*np.transpose(DCS)[1])

    def GetBSMWeights(self, PID, Energy):
        if PID == 22:
            if np.log10(Energy) < self._logEgMinDarkComp:
                return 0.0
            else:
                return self._NSigmaDarkComp(Energy)/(self._NSigmaPP(Energy) + self._NSigmaComp(Energy))
        elif PID == 11:
            if np.log10(Energy) < self._logEeMinDarkBrem:
                return 0.0
            else:
                return self._NSigmaDarkBrem(Energy)/self._NSigmaBrem(Energy)
        elif PID == -11:
            if np.log10(Energy) < self._logEeMinDarkBrem:
                BremPiece = 0.0
            else:
                BremPiece = self._NSigmaDarkBrem(Energy)
            if np.log10(Energy) < self._logEeMinDarkAnn:
                AnnPiece = 0.0
            else:
                AnnPiece = self._NSigmaDarkAnn(Energy)
            return (BremPiece + AnnPiece)/(self._NSigmaBrem(Energy) + self._NSigmaAnn(Energy))

    def GetPositronDarkBF(self, Energy):
        if np.log10(Energy) < self._logEeMinDarkAnn:
            return 1.0
        else:
            return self._NSigmaDarkBrem(Energy)/(self._NSigmaDarkBrem(Energy) + self._NSigmaDarkAnn(Energy))

    def DarkElecBremSample(self, Elec0):
        Ee0, pex0, pey0, pez0 = Elec0.get_pf()

        ThZ = np.arccos(pez0/np.sqrt(pex0**2 + pey0**2 + pez0**2))
        PhiZ = np.arctan2(pey0, pex0)
        RM = [[np.cos(ThZ)*np.cos(PhiZ), -np.sin(PhiZ), np.sin(ThZ)*np.cos(PhiZ)],
            [np.cos(ThZ)*np.sin(PhiZ), np.cos(PhiZ), np.sin(ThZ)*np.sin(PhiZ)],
            [-np.sin(ThZ), 0, np.cos(ThZ)]]

        LUKey = int((np.log10(Ee0) - self._logEeMinDarkBrem)/self._logEeSSDarkBrem)
        ts = self.get_DarkBremSamples(LUKey)
        SampEvt = ts[np.random.randint(0, len(ts))]
        EeMod = self._EeVecDarkBrem[LUKey]

        ct = np.cos(self.get_MV()/(SampEvt[0]*Ee0/EeMod)*np.sqrt((EeMod-SampEvt[0])/EeMod)*SampEvt[1])
        ctp =np.cos(self.get_MV()/(SampEvt[0]*Ee0/EeMod)*np.sqrt(EeMod/(EeMod-SampEvt[0]))*SampEvt[2])
        NFVs = eeVFourVecs(Ee0, me, SampEvt[0]*Ee0/EeMod, self.get_MV(), ct, ctp, SampEvt[3])

        EVf, pVxfZF, pVyfZF, pVzfZF = NFVs[2]
        pV3ZF = [pVxfZF, pVyfZF, pVzfZF]    
        pV3LF = np.dot(RM, pV3ZF)

        if EVf > Ee0:
            print("---------------------------------------------")
            print("High Energy V Found from Electron Samples:")
            print(Elec0.get_pf())
            print(EVf)
            print(SampEvt)
            print(LUKey)
            print(EeMod)
            print("---------------------------------------------")

        wg = self.GetBSMWeights(11, Ee0)

        GenType = 0

        NewV = Particle(4900022, EVf, pV3LF[0], pV3LF[1], pV3LF[2], Elec0.get_rf()[0], Elec0.get_rf()[1], Elec0.get_rf()[2], 2*(Elec0.get_IDs()[1])+1, Elec0.get_IDs()[1], Elec0.get_IDs()[0], Elec0.get_IDs()[4]+1, GenType, wg)
        return NewV

    def DarkAnnihilationSample(self, Elec0):
        Ee0, pex0, pey0, pez0 = Elec0.get_pf()

        ThZ = np.arccos(pez0/np.sqrt(pex0**2 + pey0**2 + pez0**2))
        PhiZ = np.arctan2(pey0, pex0)
        RM = [[np.cos(ThZ)*np.cos(PhiZ), -np.sin(PhiZ), np.sin(ThZ)*np.cos(PhiZ)],
            [np.cos(ThZ)*np.sin(PhiZ), np.cos(PhiZ), np.sin(ThZ)*np.sin(PhiZ)],
            [-np.sin(ThZ), 0, np.cos(ThZ)]]

        LUKey = int((np.log10(Ee0) - self._logEeMinDarkAnn)/self._logEeSSDarkAnn)
        EeMod = self._EeVecDarkAnn[LUKey]
        ts = self.get_DarkAnnSamples(LUKey)
        SampEvt = ts[np.random.randint(0, len(ts))]
        #NFVs = Ann_FVs(EeMod, meT, MVT, SampEvt[0])[1]
        NFVs = Ann_FVs(Ee0, me, self.get_MV(), SampEvt[0])[1]
        GenType = 2

        EVf, pVxfZF, pVyfZF, pVzfZF = NFVs
        pV3ZF = [pVxfZF, pVyfZF, pVzfZF]    
        pV3LF = np.dot(RM, pV3ZF)
        wg = self.GetBSMWeights(-11, Ee0)

        if EVf > Ee0:
            print("---------------------------------------------")
            print("High Energy V Found from Positron Samples:")
            print(Elec0.get_pf())
            print(EVf)
            print(SampEvt)
            print(LUKey)
            print(EeMod)
            print(wg)
            print("---------------------------------------------")

        NewV = Particle(4900022, EVf, pV3LF[0], pV3LF[1], pV3LF[2], Elec0.get_rf()[0], Elec0.get_rf()[1], Elec0.get_rf()[2], 2*(Elec0.get_IDs()[1])+1, Elec0.get_IDs()[1], Elec0.get_IDs()[0], Elec0.get_IDs()[4]+1, GenType, wg)
        return NewV

    def DarkComptonSample(self, Phot0):
        Eg0, pgx0, pgy0, pgz0 = Phot0.get_pf()

        ThZ = np.arccos(pgz0/np.sqrt(pgx0**2 + pgy0**2 + pgz0**2))
        PhiZ = np.arctan2(pgy0, pgx0)
        RM = [[np.cos(ThZ)*np.cos(PhiZ), -np.sin(PhiZ), np.sin(ThZ)*np.cos(PhiZ)],
            [np.cos(ThZ)*np.sin(PhiZ), np.cos(PhiZ), np.sin(ThZ)*np.sin(PhiZ)],
            [-np.sin(ThZ), 0, np.cos(ThZ)]]

        LUKey = int((np.log10(Eg0) - self._logEgMinDarkComp)/self._logEgSSDarkComp)
        EgMod = self._EgVecDarkComp[LUKey]
        ts = self.get_DarkCompSamples(LUKey)
        SampEvt = ts[np.random.randint(0, len(ts))]
        #NFVs = Compton_FVs(EgMod, meT, MVT, SampEvt[0])[1]
        NFVs = Compton_FVs(Eg0, me, self.get_MV(), SampEvt[0])[1]

        EVf, pVxfZF, pVyfZF, pVzfZF = NFVs
        pV3ZF = [pVxfZF, pVyfZF, pVzfZF]    
        pV3LF = np.dot(RM, pV3ZF)

        wg = self.GetBSMWeights(22, Eg0)
        GenType = 3
        if EVf > Eg0:
            print("---------------------------------------------")
            print("High Energy V Found from Photon Samples:")
            print(Phot0.get_pf())
            print(EVf)
            print(SampEvt)
            print(LUKey)
            print(EgMod)
            print(wg)
            print("---------------------------------------------")

        NewV = Particle(4900022, EVf, pV3LF[0], pV3LF[1], pV3LF[2], Phot0.get_rf()[0], Phot0.get_rf()[1], Phot0.get_rf()[2], 2*(Phot0.get_IDs()[1])+0, Phot0.get_IDs()[1], Phot0.get_IDs()[0], Phot0.get_IDs()[4]+1, GenType, wg)
        return NewV

    def GenDarkShower(self, ExDir=None, SParams=None):
        if ExDir is None and SParams is None:
            print("Need an existing SM shower-file directory or SM shower parameters to run dark shower")
            return None
        
        if ExDir is not None:
            ShowerToSamp = np.load(ExDir, allow_pickle=True)
        else:
            PID0, p40, ParPID = SParams
            ShowerToSamp = self.GenShower(PID0, p40, ParPID)
        
        NewShower = []
        for ap in ShowerToSamp:
            if ap.get_IDs()[0] == 11:
                if np.log10(ap.get_pf()[0]) < self._logEeMinDarkBrem:
                    continue
                npart = self.DarkElecBremSample(ap)
            elif ap.get_IDs()[0] == -11:
                if np.log10(ap.get_pf()[0]) < self._logEeMinDarkBrem:
                    continue
                DarkBFEpBrem = self.GetPositronDarkBF(ap.get_pf()[0])
                ch = np.random.uniform(low=0., high=1.0)
                if ch < DarkBFEpBrem:
                    npart = self.DarkElecBremSample(ap)
                else:
                    npart = self.DarkAnnihilationSample(ap)
            elif ap.get_IDs()[0] == 22:
                if np.log10(ap.get_pf()[0]) < self._logEgMinDarkComp:
                    continue
                npart = self.DarkComptonSample(ap)
            NewShower.append(npart)

        return ShowerToSamp, NewShower
