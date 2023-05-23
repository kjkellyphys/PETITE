import numpy as np
from PETITE.dark_shower import DarkShower

NBP_dir = '../NBP/'
input_dir = '../Inputs/'
output_dir = './Outputs/'

MinEnergy = 0.015
sGraphite = DarkShower(NBP_dir, 'graphite', MinEnergy, '10MeV')
Parents = np.load(input_dir+"Photons_From_Pi0s_120GeV.npy")

ExDir0 = output_dir+'Particles_PionShower_Graphite.npy'
gds = sGraphite.GenDarkShower(ExDir=ExDir0, SParams=None)

np.save(output_dir+"DarkShower_PionShower_10MeV_Graphite", gds)
