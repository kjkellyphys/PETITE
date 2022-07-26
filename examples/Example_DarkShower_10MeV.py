#from MCCode import *
from PETITE.ShowerCode import *

MinEnergy = 0.015
sGraphite = DarkShower('/Users/kjkelly/Documents/GitHub/PETITE/NBP/', 'graphite', MinEnergy, '10MeV')
Parents = np.load("/Users/kjkelly/Documents/GitHub/PETITE/Inputs/Photons_From_Pi0s_120GeV.npy")

ExDir0 = '/Users/kjkelly/Documents/GitHub/PETITE/examples/Outputs/Particles_PionShower_Graphite.npy'
gds = sGraphite.GenDarkShower(ExDir=ExDir0, SParams=None)

np.save("./Outputs/DarkShower_PionShower_10MeV_Graphite", gds)