'''For internal consistency, all physical units and constants are defined here.
FIXME: describe target_information dictionary
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

m_muon = 105.6583755*MeV
m_pi0, c_tau_pi0 = 134.9768*MeV, 25.3*(1e-9) #meters
m_pi_pm, c_tau_pi_pm = 139.57039*MeV, 7.8045 #meters

m_eta, Gamma_eta = 547.862*MeV, 1.31*keV
m_eta_prime, Gamma_eta_prime = 957.78*MeV, 0.188*MeV

target_information = {"graphite": {"Z_T":6,  "A_T":12, "mT":11.178},
                      "lead"    : {"Z_T":82, "A_T":207, "mT":207.2},
                      "hydrogen": {"Z_T":1,  "A_T":1,   "mT":1.0}}