import numpy as np

class Particle:
    """Container for particle information as it is propagated through target
    """
    def __init__(self, PID, E0, px0, py0, pz0, x0, y0, z0, ID, ParID, ParPID, GenID, GenProcess, Weight):
        """Initializes an instance of the Particle class
        Args:
            PID: PDG id of the particle
            E0, px0, py0, pz0: energy and components of momentum
            x0, y0, z0: current coordinates of the particle in the target
            ID: a unique label of the particle in the shower
            ParID: shower label of the parent particle
            ParPID: PGD id of the parent paricle
            GenID: 
            GenProcess: id of the process that produced the particle
            Weight: probability weight for the process that generated this particle
        """
        self.set_IDs(np.array([PID, ID, ParPID, ParID, GenID, GenProcess, Weight]))

        self.set_p0(np.array([E0, px0, py0, pz0]))
        self.set_r0(np.array([x0, y0, z0]))

        self.set_Ended(False)

        self.set_pf(np.array([E0,px0,py0,pz0]))
        self.set_rf(np.array([x0, y0, z0]))

    def set_IDs(self, value):
        self._IDs = value
    def get_IDs(self):
        return self._IDs

    def set_p0(self, value):
        self._p0 = value
    def get_p0(self):
        return self._p0
    def set_pf(self, value):
        self._pf = value
    def get_pf(self):
        return self._pf

    def set_r0(self, value):
        self._r0 = value
    def get_r0(self):
        return self._r0
    def set_rf(self, value):
        self._rf = value
    def get_rf(self):
        return self._rf

    def set_Ended(self, value):    
        if value != True and value != False:
            raise ValueError("Ended property must be a boolean.")
        self._Ended = value
    def get_Ended(self):
        return self._Ended

