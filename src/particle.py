import numpy as np

default_ids = {"PID":11, "ID":0, "parent_PID":22, "parent_ID":-1, "generation_number":0, "generation_process":"Input", "weight":1.0, "mass":None}
class Particle:
    """Container for particle information as it is propagated through target
    """
#    def __init__(self, PID, E0, px0, py0, pz0, x0, y0, z0, ID, ParID, ParPID, GenID, GenProcess, Weight, mass=None):
    def __init__(self, p0, r0, id_dictionary=None):
        """Initializes an instance of the Particle class
        Args:
            PID: PDG id of the particle
            p0: (four-vector) energy and components of momentum
            r0: (three-vector) current coordinates of the particle in the target
            ID: a unique label of the particle in the shower

            id_dictionary: dictionary containing following identification keys (defaults given for unspecified information):
                --PID (PDG particle ID) -- default:11
                --ID (ID for shower development) default:0
                --parent_PID (parent-particle's PDG ID) -- default:22
                --parent_ID (shower-ID of parent particle) -- default:-1
                --generation_number (number of splittings before this particle was created) -- default:0
                --generation_process (string denoting process by which this particle was created) -- default:"Input"
                --weight (used for dark-particle generation for weighted showers) -- default:1
                --mass (mass of the particle) -- default:None (gets set later)
            ParID: shower label of the parent particle
            ParPID: PGD id of the parent paricle
            GenID: 
            GenProcess: id of the process that produced the particle
            Weight: probability weight for the process that generated this particle
        """

        #self.set_ids(np.array([PID, ID, ParPID, ParID, GenID, GenProcess, Weight]))
        if id_dictionary is None:
            id_dictionary = {}
        self.set_ids(id_dictionary)

        self.set_mass(self.get_ids()['mass'])
        self.set_p0(p0)
        self.set_r0(r0)

        self.set_ended(False)

        self.set_pf(p0)
        self.set_rf(r0)

    def set_ids(self, value):
        self._IDs = {}
        for key in default_ids.keys():
            if key in value.keys():
                self._IDs[key] = value[key]
            else:
                self._IDs[key] = default_ids[key]
    #def set_ids(self, value):
    #    self._IDs = value
    def get_ids(self):
        return self._IDs

    def get_pid(self):
        """Returns PID of particle in shower
        """
        return self._IDs["PID"]
    def get_parent_pid(self):
        """Returns PID of particle's parent in shower
        """
        return self._IDs["parent_PID"]
    def get_weight(self):
        """Returns weight of particle in shower
        """
        return self._IDs["weight"]

    def set_mass(self, value):
        self._mass = value
    def set_p0(self, value):
        self._p0 = value
        invariant_mass = round(np.sqrt(round(value[0]**2 - value[1]**2 - value[2]**2 - value[3]**2, 12)),6)
        if self._mass is not None: #Check for proper definition of invariant mass
            if invariant_mass != round(self._mass,6):
                print("Error setting mass of new particle")
                print(self._mass, invariant_mass)
        else: #If mass is not provided, set it here
            self._mass = invariant_mass
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

    def set_ended(self, value):    
        if value != True and value != False:
            raise ValueError("Ended property must be a boolean.")
        self._Ended = value
    def get_ended(self):
        return self._Ended

    def rotation_matrix(self):
        """
        Determines the rotation matrix between the z-axis and the particle's (final) three-momentum
        """
        E0, px0, py0, pz0 = self.get_pf()
        ThZ = np.arccos(pz0/np.sqrt(px0**2 + py0**2 + pz0**2))
        PhiZ = np.arctan2(py0, px0)
        return [[np.cos(ThZ)*np.cos(PhiZ), -np.sin(PhiZ), np.sin(ThZ)*np.cos(PhiZ)],
            [np.cos(ThZ)*np.sin(PhiZ), np.cos(PhiZ), np.sin(ThZ)*np.sin(PhiZ)],
            [-np.sin(ThZ), 0, np.cos(ThZ)]]

    def boost_matrix(self):
        """
        Determines the boost matrix between the particle's rest-frame and lab-frame*
        """
        E0, px0, py0, pz0 = self.get_pf()
        m0 = self._mass

        gamma = E0/m0
        beta = np.sqrt(1.0 - 1.0/gamma**2)
        betax, betay, betaz = beta*np.array([px0, py0, pz0])/np.linalg.norm([px0, py0, pz0])

        return [[gamma, gamma*betax, gamma*betay, gamma*betaz],
                [gamma*betax, 1 + (gamma-1)*betax**2, (gamma-1)*betax*betay, (gamma-1)*betax*betaz],
                [gamma*betay, (gamma-1)*betay*betax, 1 + (gamma-1)*betay**2, (gamma-1)*betay*betaz],
                [gamma*betaz, (gamma-1)*betaz*betax, (gamma-1)*betaz*betay, 1 + (gamma-1)*betaz**2]]

    def two_body_decay(self, decay_masses, angular_information="Isotropic"):
        mX = self._mass
        m1, m2 = decay_masses

        E1 = (mX**2 - m2**1 + m1**2)/(2*mX)
        E2 = (mX**2 - m1**1 + m2**2)/(2*mX)
        pF = np.sqrt(E1**2 - m1**2)
        if angular_information == "Isotropic":
            cos_theta = np.random.uniform(-1.0, 1.0)
            phi = np.random.uniform(0.0, 2.0*np.pi)
        elif len(angular_information) == 2:
            cos_theta_c, phi_c = np.random.uniform(low=0.0, high=1.0, size=2)
            cos_theta = angular_information[0](cos_theta_c)
            phi = angular_information[1](phi_c)
        else: #If two functions are not given, assume the one given is for cos(theta), phi is uniform
            cos_theta_c = np.random.uniform(low=0.0, high=1.0, size=1)
            cos_theta = angular_information(cos_theta_c)
            phi = np.random.uniform(0.0, 2.0*np.pi)
        sin_theta = np.sqrt(1 - cos_theta**2)

        p1_four_vector_RF = [E1, pF*sin_theta*np.sin(phi), pF*sin_theta*np.cos(phi), pF*cos_theta]
        p2_four_vector_RF = [E2, -pF*sin_theta*np.sin(phi), -pF*sin_theta*np.cos(phi), -pF*cos_theta]
        RM = self.boost_matrix()
        p1_four_vector_LF = np.dot(RM, p1_four_vector_RF)
        p2_four_vector_LF = np.dot(RM, p2_four_vector_RF)

        new_particle_1 = Particle(4900023, p1_four_vector_LF[0])

    def decay_particle(self, decay_product_masses, decay_type="TwoBody"):
        if len(decay_product_masses) < 2:
            raise ValueError("Decay into fewer than two final-state particles called.")
        if np.sum(decay_product_masses) > self._mass:
            raise ValueError("Decay into particles with too great msas called.")
        