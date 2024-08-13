'''For internal consistency, all physical units and constants are defined here.
'''

GeV = 1
MeV = 1E-3
keV = 1E-6
eV = 1E-9
centimeter = 1
##  https://pdg.lbl.gov/2022/reviews/contents_sports.html
alpha_em = 1/137.035999
m_electron = 510.998950*keV
m_proton = 938.272088*MeV
m_proton_grams = 1.67262192369E-24
m_neutron = 939.56542052*MeV
atomic_mass_unit = 931.49410241*MeV
n_avogadro = 6.0221409e+23
hbarc = 0.1973269804E-13*GeV*centimeter
GeVsqcm2 = hbarc**2
cmtom = 0.01

m_muon = 105.6583755*MeV
m_tau = 1776.93*MeV
m_pi0, c_tau_pi0 = 134.9768*MeV, 25.3*(1e-9) #meters
m_pi_pm, c_tau_pi_pm = 139.57039*MeV, 7.8045 #meters

m_eta, Gamma_eta = 547.862*MeV, 1.31*keV
m_eta_prime, Gamma_eta_prime = 957.78*MeV, 0.188*MeV

"""
target_information is a dictionary that is loaded in throughout the code
if additional targets are desired, please add them here
Required information:
    Z_T: atomic number of target
    A_T: atomic mass of target
    mT: mass of target in GeV (needed for dark shower production)
    rho: density of target in g/cm^3
"""
target_information = {"graphite": {"Z_T":6,  "A_T":12, "mT":11.178, "rho":2.210},
                      "lead"    : {"Z_T":82, "A_T":207, "mT":207.2, "rho":11.35},
                      "iron"    : {"Z_T":26, "A_T":56,  "mT":55.845, "rho":8.00},
                      "hydrogen": {"Z_T":1,  "A_T":1,   "mT":1.0, "rho":1.0},
                      "aluminum": {"Z_T":13, "A_T":27,  "mT":26.9815385, "rho":2.699},
                      "tungsten": {"Z_T":74, "A_T":183.84, "mT":183.84, "rho":19.3},
                      "molybdenum":{"Z_T":42, "A_T":95.95, "mT":95.95, "rho":10.2}}

for tm in target_information:
    target_information[tm]['dEdx'] = 2.0*target_information[tm]['rho'] # MeV/cm