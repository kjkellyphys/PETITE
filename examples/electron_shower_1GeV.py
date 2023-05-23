import numpy as np
from PETITE.shower import Shower
meT = 0.000511

NSamp = 100
Ee0 = 1.0
Mom40 = [Ee0, 0.0, 0.0, np.sqrt(Ee0**2 - meT**2)]
MinEnergy = 0.002

NBP_dir = '../NBP/'
output_dir = './Outputs/'

sLead = Shower(NBP_dir, 'lead', MinEnergy)
s0 = sLead.generate_shower(11, Mom40, 0, VB=True)
ts0 = np.array([np.concatenate([s5i.get_r0(), s5i.get_rf(), [s5i.get_ids()[0]]]) for s5i in s0])
np.save(output_dir + "AllParticles_SingleShower_GeV_Lead", ts0)

s5all = np.concatenate([sLead.generate_shower(11, Mom40, 0) for ni in range(NSamp)])
ts = np.array([np.concatenate([s5i.get_r0(), s5i.get_rf(), s5i.get_p0(), s5i.get_pf(), s5i.get_IDs()]) for s5i in s5all])
np.save(output_dir + "AllParticles_GeVShower_Lead", ts)

sGraphite = Shower(NBP_dir, 'graphite', MinEnergy)
s0 = sGraphite.generate_shower(11, Mom40, 0)
ts0 = np.array([np.concatenate([s5i.get_r0(), s5i.get_rf(), [s5i.get_ids()[0]]]) for s5i in s0])
np.save(output_dir + "AllParticles_SingleShower_GeV_Graphite", ts0)

s5all = np.concatenate([sGraphite.generate_shower(11, Mom40, 0) for ni in range(NSamp)])
ts = np.array([np.concatenate([s5i.get_r0(), s5i.get_rf(), s5i.get_p0(), s5i.get_pf(), s5i.get_IDs()]) for s5i in s5all])
np.save(output_dir + "AllParticles_GeVShower_Graphite", ts)
