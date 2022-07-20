from MCCode import *

APSaves = []
meT = 0.000511

NSamp = 100
Ee0 = 1.0
Mom40 = [Ee0, 0.0, 0.0, np.sqrt(Ee0**2 - meT**2)]

s5all = np.concatenate([Shower(11, Mom40, 0) for ni in range(NSamp)])
ts = np.array([np.array([s5i.x0, s5i.y0, s5i.z0, s5i.xf, s5i.yf, s5i.zf, s5i.E0, s5i.px0, s5i.py0, s5i.pz0, s5i.Ef, s5i.pxf, s5i.pyf, s5i.pzf, s5i.PID, s5i.ID, s5i.ParID, s5i.ParPID, s5i.GenProcess, s5i.GenID]) for s5i in s5all])
np.save("AllParticles_GeVShower_Graphite", ts)