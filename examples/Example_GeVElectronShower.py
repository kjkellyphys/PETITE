#from MCCode import *
from PETITE.ShowerCode import *
meT = 0.000511

NSamp = 5000
Ee0 = 1.0
Mom40 = [Ee0, 0.0, 0.0, np.sqrt(Ee0**2 - meT**2)]

sLead = Shower('/Users/kjkelly/Documents/GitHub/PETITE/NBP/', 'lead')
s0 = sLead.GenShower(11, Mom40, 0)
ts0 = np.array([np.concatenate([s5i.get_r0(), s5i.get_rf(), [s5i.get_IDs()[0]]]) for s5i in s0])
np.save("./Outputs/AllParticles_SingleShower_GeV_Lead", ts0)

s5all = np.concatenate([sLead.GenShower(11, Mom40, 0) for ni in range(NSamp)])
ts = np.array([np.concatenate([s5i.get_r0(), s5i.get_rf(), s5i.get_p0(), s5i.get_pf(), s5i.get_IDs()]) for s5i in s5all])
np.save("./Outputs/AllParticles_GeVShower_Lead", ts)

sGraphite = Shower('/Users/kjkelly/Documents/GitHub/PETITE/NBP/', 'graphite')
s0 = sGraphite.GenShower(11, Mom40, 0)
ts0 = np.array([np.concatenate([s5i.get_r0(), s5i.get_rf(), [s5i.get_IDs()[0]]]) for s5i in s0])
np.save("./Outputs/AllParticles_SingleShower_GeV_Graphite", ts0)

s5all = np.concatenate([sGraphite.GenShower(11, Mom40, 0) for ni in range(NSamp)])
ts = np.array([np.concatenate([s5i.get_r0(), s5i.get_rf(), s5i.get_p0(), s5i.get_pf(), s5i.get_IDs()]) for s5i in s5all])
np.save("./Outputs/AllParticles_GeVShower_Graphite", ts)