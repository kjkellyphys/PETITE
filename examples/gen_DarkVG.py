from PETITE.AllProcesses import *
import numpy as np
import os
import random as rnd
import itertools
from multiprocessing import Pool
import pickle

DarkVMass = '30MeV'
MVT = 0.030

meT, ZT, alT, EgMinT = 0.000511, 26.0, 1.0/137.0, 0.001
E0Brem = 1.1*MVT
E0Comp = 1.05*(MVT**2/(2*meT) + MVT)
E0Ann = 1.05*(MVT**2/(2*meT) - meT)

EnergyVecBrem = np.logspace(np.log10(E0Brem), np.log10(100.0), 100)
EnergyVecComp = np.logspace(np.log10(E0Comp), np.log10(100.0), 100)
EnergyVecAnn = np.logspace(np.log10(E0Ann), np.log10(100.0), 100)
kEnergyVec = [k for k in range(len(EnergyVecComp))]

Dir0 = os.getcwd()

svDir = Dir0 + "/DarkBremPickes_TMP/"
if os.path.exists(svDir) == False:
    os.system("mkdir " + svDir)
svBrem = svDir + "BremPickles_" + DarkVMass + "_"
svAnn = svDir + "AnnPickles_" + DarkVMass + "_"
svCom = svDir + "ComptonPickles_" + DarkVMass + "_"

def RetSamp(theta):
    kEi = theta[0]
    strsaveB = svBrem + str(kEi) + ".p"
    strsaveA = svAnn + str(kEi) + ".p"
    strsaveC = svCom + str(kEi) + ".p"

    if os.path.exists(strsaveA):
        print("Already completed this point for Annihilation")
        tr = 1.0
    else:
        EiAnn = EnergyVecAnn[kEi]
        s0 = Ann_S([EiAnn, meT, alT, MVT], EgMinT, VB=False, mode="Pickle")
        pickle.dump(s0, open(strsaveA, "wb"))

        tr = 2.0
    if os.path.exists(strsaveC):
        print("Already completed this point for Compton")
        tr = 3.0
    else:
        EiComp = EnergyVecComp[kEi]
        
        s0c = Compton_S([EiComp, meT, MVT, alT], False, mode="Pickle")
        pickle.dump(s0c, open(strsaveC, "wb"))
        tr = 4.0
    if os.path.exists(strsaveB):
        print("Already completed this point for Bremsstrahlung")
        tr = 5.0
    else:
        Ei = EnergyVecBrem[kEi]

        s0 = DBrem_S_T([Ei, meT, MVT, ZT, alT], False, mode="Pickle")
        pickle.dump(s0, open(strsaveB, "wb"))
        tr = 6.0

    return tr

paramlist = list(itertools.product(kEnergyVec))
if __name__ == '__main__':
    #pool = Pool()
    #res = pool.map(RetSamp, paramlist)

    ts = np.array([pickle.load(open(svAnn+str(ki)+".p", "rb")) for ki in range(100)])
    np.save("../NBP/DarkV/AnnihilationPickles_"+DarkVMass, np.transpose([EnergyVecAnn, ts]))

    ts = np.array([pickle.load(open(svCom+str(ki)+".p", "rb")) for ki in range(100)])
    np.save("../NBP/DarkV/ComptonPickles_"+DarkVMass, np.transpose([EnergyVecComp, ts]))

    ts = np.array([pickle.load(open(svBrem+str(ki)+".p", "rb")) for ki in range(100)])
    np.save("../NBP/DarkV/ElectronPositron_BremPickles_"+DarkVMass, np.transpose([EnergyVecBrem, ts]))

    #Cleanup
    for ki in range(100):
        os.system("rm " + svAnn + str(ki) + ".p")
        os.system("rm " + svCom + str(ki) + ".p")
        os.system("rm " + svBrem + str(ki) + ".p")