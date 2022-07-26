#from MCCode import *
from PETITE.ShowerCode import *

MinEnergy = 0.015
sGraphite = Shower('/Users/kjkelly/Documents/GitHub/PETITE/NBP/', 'graphite', MinEnergy)
Parents = np.load("/Users/kjkelly/Documents/GitHub/PETITE/Inputs/Photons_From_Pi0s_120GeV.npy")

NSamp = 500
s5all = np.concatenate([sGraphite.GenShower(22, Parents[np.random.randint(0, len(Parents))], 111, VB=(np.mod(ni,50)==0)) for ni in range(NSamp)])
np.save("./Outputs/Particles_PionShower_Graphite", s5all)
#ts = np.array([np.concatenate([s5i.get_r0(), s5i.get_rf(), s5i.get_p0(), s5i.get_pf(), s5i.get_IDs()]) for s5i in s5all])
#np.save("./Outputs/AllParticles_PionShower_Graphite", ts)

'''
LDet0, RDet0 = 574.0, 2.5

def RFront(part, LDet):
    x, y, z = part.get_r0()
    E, px, py, pz = part.get_p0()
    return np.sqrt((x + px/pz*(LDet-z))**2 + (y + py/pz*(LDet-z))**2)

ts = []
NPhotonsOverall, NPassOverall = 0, 0
for ni in range(NSamp):
    s5 = sGraphite.GenShower(22, Parents[np.random.randint(0, len(Parents))], 111, VB=False)
    NPhotons, NPass = 0, 0
    for s5i in s5:
        if s5i.get_IDs()[0] == 22:
            NPhotons += 1
            NPhotonsOverall += 1
            if RFront(s5i, LDet0) <= RDet0:
                NPass += 1
                NPassOverall += 1
                ts.append(np.concatenate([s5i.get_r0(), s5i.get_rf(), s5i.get_p0(), s5i.get_pf(), s5i.get_IDs()]))
    if np.mod(ni, 10) == 0:
        print(ni, len(s5), NPhotons, NPass, NPass/NPhotons, NPhotonsOverall, NPassOverall, NPassOverall/NPhotonsOverall)
np.save("./Outputs/PassingPhotons_PionShower_Graphite", ts)
'''