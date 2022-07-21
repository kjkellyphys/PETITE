#from MCCode import *
from PETITE.ShowerCode import *

MinEnergy = 0.002
sGraphite = Shower('/Users/kjkelly/Documents/GitHub/PETITE/NBP/', 'graphite', MinEnergy)
Parents = np.load("/Users/kjkelly/Documents/GitHub/PETITE/Inputs/Photons_From_Pi0s_120GeV.npy")

NSamp = 500
s5all = np.concatenate([sGraphite.GenShower(11, Parents[np.random.randint(0, len(Parents))], 111) for ni in range(NSamp)])
ts = np.array([np.concatenate([s5i.get_r0(), s5i.get_rf(), s5i.get_p0(), s5i.get_pf(), s5i.get_IDs()]) for s5i in s5all])
np.save("./Outputs/AllParticles_PionShower_Graphite", ts)