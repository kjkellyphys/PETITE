import numpy as np
from .physical_constants import *

mass_dict = {11: m_electron, -11:m_electron,
             22:0.0, 13:m_muon, -13:m_muon,
             111:m_pi0, 211:m_pi_pm, -211:m_pi_pm,
             221:m_eta, 331:m_eta_prime, 2212:m_proton,
             12:0.0, -12:0.0, 14:0.0, -14:0.0}

default_ids = {"PID":11, "ID":1, "parent_PID":22, 
               "parent_ID":-1, "generation_number":0, 
               "generation_process":"Input", "weight":1.0, 
               "mass":None, "stability":"stable",
               "production_time":0.0,
               "decay_time":0.0,
               "interaction_time":0.0}

#pi0 (111) decays to gamma gamma with Br = 0.98823
#eta (221) decays to gamma gamma with Br = 0.3936
#                 to three pi0 with Br = 0.3257

#eta-prime (331) decays to gamma gamma with Br = 0.02307
#                       to pi0 + pi0 + eta with Br = 0.224
#                       to three pi0 with Br = 0.00250
meson_decay_dict = {111: [[0.98823, [22,22]]],
                    221: [[0.3936, [22,22]], [0.3257, [111, 111, 111]]],
                    331: [[0.02307, [22,22]], [0.224, [111, 111, 221]], [0.00250, [111, 111, 111]]]}
meson_twobody_branchingratios = {pid0:meson_decay_dict[pid0][0][0] for pid0 in meson_decay_dict.keys()}

class Particle:
    """Container for particle information as it is propagated through target
    """
    def __init__(self, p0, r0=np.array([0,0,0]), id_dictionary=None):
        """Initializes an instance of the Particle class
        Args:
            PID: PDG id of the particle
            p0: (four-vector) energy and components of momentum
            r0: (three-vector) beginning coordinates of the particle in the target
                -default:origin

            id_dictionary: dictionary containing following identification keys (defaults given for unspecified information):
                --PID (PDG particle ID) -- default:11
                --ID (ID for shower development) default:0
                --parent_PID (parent-particle's PDG ID) -- default:22
                --parent_ID (shower-ID of parent particle) -- default:-1
                --generation_number (number of splittings before this particle was created) -- default:0
                --generation_process (string denoting process by which this particle was created) -- default:"Input"
                --weight (used for dark-particle generation for weighted showers) -- default:1
                --mass (mass of the particle) -- default:None (gets set later)
                --stability (string identifying whether particle is stable/short-lived/long-lived) -- default:"stable"
        """

        if id_dictionary is None:
            id_dictionary = {}
        self.set_ids(id_dictionary)

        self.set_mass(self.get_ids()['mass'])
        if type(p0) is list:
            p0 = np.array(p0)
        #if p0 is given as a number, assume it to be the particle's energy,
        #momentum pointing in z-direction
        elif type(p0) is int or type(p0) is float:
            if self.get_ids()["mass"] is None:
                self._mass = mass_dict[self.get_ids()["PID"]]
            p0 = np.array([p0, 0, 0, np.sqrt(p0**2 - self._mass**2)])
        self.set_p0(p0)
        if type(r0) is list:
            r0 = np.array(r0)
        self.set_r0(r0)

        # the ended key is used to determine whether the particle is an intermediate particle in the shower (False) or a final particle (True)
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
    def get_ids(self):
        return self._IDs
    
    def update_ids(self, key, value):
        self._IDs[key] = value

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
        if self._mass is None:
            #else: #If mass is not provided, set it here
            invariant_mass = round(np.sqrt(round(value[0]**2 - value[1]**2 - value[2]**2 - value[3]**2, 12)),6)
            self._mass = invariant_mass
            self._IDs["mass"] = invariant_mass
    def get_p0(self):
        return self._p0
    def set_pf(self, value):
        self._pf = value
    def get_pf(self):
        return self._pf
    
    def get_angle_to_z_0(self):
        E0, px0, py0, pz0 = self.get_p0()
        return np.arccos(pz0/np.sqrt(px0**2 + py0**2 + pz0**2))

    def lose_energy(self, value):
        E0, px0, py0, pz0 = self.get_pf()
        p30 = np.linalg.norm([px0, py0, pz0])
        E_updated = E0 - value
        if E_updated < self.get_ids()["mass"]:
            E_updated = self.get_ids()["mass"]
        p3f = np.sqrt(E_updated**2 - self.get_ids()["mass"]**2)
        if p3f > 0.0:
            self.set_pf([E_updated, px0/p30*p3f, py0/p30*p3f, pz0/p30*p3f])

    def set_r0(self, value):
        self._r0 = value
    def get_r0(self):
        return self._r0
    def set_rf(self, value):
        self._rf = value
    def get_rf(self):
        return self._rf

    def set_ended(self, value):    
        '''Sets the ended property of the particle. If True, the particle is a final particle in the shower.'''
        if value != True and value != False:
            raise ValueError("Ended property must be a boolean.")
        self._Ended = value

    def get_ended(self):
        return self._Ended

    def copy(self):
        return Particle(self.get_p0(), self.get_r0(), self.get_ids())

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
        Determines the boost matrix between the particle's rest-frame and lab-frame
        """
        E0, px0, py0, pz0 = self.get_pf()
        m0 = self._mass

        gamma = E0/m0
        beta = np.sqrt(1.0 - 1.0/gamma**2)
        pmag = np.linalg.norm([px0, py0, pz0])
        if pmag == 0.0:
            return np.identity(4)
        betax, betay, betaz = beta*np.array([px0, py0, pz0])/pmag

        return [[gamma, gamma*betax, gamma*betay, gamma*betaz],
                [gamma*betax, 1 + (gamma-1)*betax**2/beta**2, (gamma-1)*betax*betay/beta**2, (gamma-1)*betax*betaz/beta**2],
                [gamma*betay, (gamma-1)*betay*betax/beta**2, 1 + (gamma-1)*betay**2/beta**2, (gamma-1)*betay*betaz/beta**2],
                [gamma*betaz, (gamma-1)*betaz*betax/beta**2, (gamma-1)*betaz*betay/beta**2, 1 + (gamma-1)*betaz**2/beta**2]]

    def two_body_decay(self, p1_dict, p2_dict, angular_information="Isotropic"):
        mX = self._mass
        if ("mass") not in p1_dict.keys():
            if ("PID") not in p1_dict.keys():
                raise ValueError("Masses must be included in `p1_dict' when calling two_body_decay()")
            else:
                try:
                    p1_dict['mass'] = mass_dict[p1_dict["PID"]]
                except:
                    raise ValueError("PID provided for unspecified particle.")
        if ("mass") not in p2_dict.keys():
            if ("PID") not in p2_dict.keys():
                raise ValueError("Masses must be included in `p2_dict' when calling two_body_decay()")
            else:
                try:
                    p2_dict['mass'] = mass_dict[p2_dict["PID"]]
                except:
                    raise ValueError("PID provided for unspecified particle.")

        m1, m2 = p1_dict['mass'], p2_dict['mass']

        E1 = (mX**2 - m2**2 + m1**2)/(2*mX)
        E2 = (mX**2 - m1**2 + m2**2)/(2*mX)
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

        #Includes the factor of g_{mu nu} so that we can use numpy's build in .dot() function
        p1_four_vector_RF = [E1, -pF*sin_theta*np.sin(phi), -pF*sin_theta*np.cos(phi), -pF*cos_theta]
        p2_four_vector_RF = [E2, pF*sin_theta*np.sin(phi), pF*sin_theta*np.cos(phi), pF*cos_theta]
        boost = self.boost_matrix()
        p1_four_vector_LF = np.dot(boost, p1_four_vector_RF)
        p2_four_vector_LF = np.dot(boost, p2_four_vector_RF)

        new_particle_1 = Particle(p1_four_vector_LF, self.get_rf(), p1_dict)
        new_particle_2 = Particle(p2_four_vector_LF, self.get_rf(), p2_dict)
        
        return [new_particle_1, new_particle_2]
    
    def dalitz_range(self, m12sq, m1, m2, m3, M):
        #returns the allowed range for m23sq given m12sq
        E2s = (m12sq - m1**2 + m2**2)/(2*np.sqrt(m12sq))
        E3s = (M**2 - m12sq - m3**2)/(2*np.sqrt(m12sq))
        m23sqmin = (E2s + E3s)**2 - (np.sqrt(E2s**2 - m2**2) + np.sqrt(E3s**2 - m3**2))**2
        m23sqmax = (E2s + E3s)**2 - (np.sqrt(E2s**2 - m2**2) - np.sqrt(E3s**2 - m3**2))**2
        return m23sqmin, m23sqmax

    def three_body_decay(self, p1_dict, p2_dict, p3_dict, dalitz_information="Flat", angular_information="Isotropic"):
        mX = self._mass
        if ("mass") not in p1_dict.keys():
            if ("PID") not in p1_dict.keys():
                raise ValueError("Masses must be included in `p1_dict' when calling three_body_decay()")
            else:
                try:
                    p1_dict['mass'] = mass_dict[p1_dict["PID"]]
                except:
                    raise ValueError("PID provided for unspecified particle.")
        if ("mass") not in p2_dict.keys():
            if ("PID") not in p2_dict.keys():
                raise ValueError("Masses must be included in `p2_dict' when calling three_body_decay()")
            else:
                try:
                    p2_dict['mass'] = mass_dict[p2_dict["PID"]]
                except:
                    raise ValueError("PID provided for unspecified particle.")
        if ("mass") not in p3_dict.keys():
            if ("PID") not in p3_dict.keys():
                raise ValueError("Masses must be included in `p3_dict' when calling three_body_decay()")
            else:
                try:
                    p3_dict['mass'] = mass_dict[p3_dict["PID"]]
                except:
                    raise ValueError("PID provided for unspecified particle.")

        m1, m2, m3 = p1_dict['mass'], p2_dict['mass'], p3_dict['mass']
        if dalitz_information != "Flat":
            raise ValueError("Matrix-element-level Dalitz sampling not yet implemented")
        m12sq_min = (m1 + m2)**2
        m12sq_max = (mX - m3)**2
        m23sq_min = (m2 + m3)**2
        m23sq_max = (mX - m1)**2
        sample_found = False
        while sample_found is False:
            m12sq, m23sq = np.random.uniform(m12sq_min, m12sq_max), np.random.uniform(m23sq_min, m23sq_max)
            dr = self.dalitz_range(m12sq, m1, m2, m3, mX)
            if m23sq > dr[0] and m23sq < dr[1]:
                sample_found = True

        E1 = (mX**2 - m23sq + m1**2)/(2*mX)
        E3 = (mX**2 - m12sq + m3**2)/(2*mX)
        E2 = mX - E1 - E3
        p1, p2, p3 = np.sqrt(E1**2 - m1**2), np.sqrt(E2**2 - m2**2), np.sqrt(E3**2 - m3**2)

        #opening angle between p1 and p2 three-vectors in X rest-frame
        cos_theta_12 = (m1**2 + m2**2 + 2*E1*E2 - m12sq)/(2*p1*p2)
        sin_theta_12 = np.sqrt(1 - cos_theta_12**2)

        if angular_information == "Isotropic":
            cos_theta = np.random.uniform(-1.0, 1.0)
            phi, gamma = np.random.uniform(0.0, 2.0*np.pi, size=2)
        elif len(angular_information) == 3:
            cos_theta_c, phi_c, gamma_c = np.random.uniform(low=0.0, high=1.0, size=3)
            cos_theta = angular_information[0](cos_theta_c)
            phi = angular_information[1](phi_c)
            gamma = angular_information[2](gamma_c)
        else: #If three functions are not given, assume the one given is for cos(theta), phi is uniform
            cos_theta_c = np.random.uniform(low=0.0, high=1.0, size=1)
            cos_theta = angular_information(cos_theta_c)
            phi, gamma = np.random.uniform(0.0, 2.0*np.pi, size=2)
        sin_theta = np.sqrt(1 - cos_theta**2)

        p1_four_vector_RF = [E1, -p1*sin_theta*np.sin(phi), -p1*sin_theta*np.cos(phi), -p1*cos_theta]
        p2_four_vector_RF = [E2, p2*(sin_theta_12*np.cos(phi)*np.sin(gamma) - sin_theta_12*np.cos(gamma)*cos_theta*np.sin(phi) - cos_theta_12*sin_theta*np.sin(phi)),
                                 p2*(-np.cos(phi)*(sin_theta_12*np.cos(gamma)*cos_theta + cos_theta_12*sin_theta) - sin_theta_12*np.sin(gamma)*np.sin(phi)),
                                 p2*(sin_theta_12*np.cos(gamma)*sin_theta - cos_theta_12*cos_theta)]
        p3_four_vector_RF = [E3, -p1_four_vector_RF[1] - p2_four_vector_RF[1], -p1_four_vector_RF[2] - p2_four_vector_RF[2], -p1_four_vector_RF[3] - p2_four_vector_RF[3]]

        boost = self.boost_matrix()
        
        p1_four_vector_LF = np.dot(boost, p1_four_vector_RF)
        p2_four_vector_LF = np.dot(boost, p2_four_vector_RF)
        p3_four_vector_LF = np.dot(boost, p3_four_vector_RF)

        new_particle_1 = Particle(p1_four_vector_LF, self.get_rf(), p1_dict)
        new_particle_2 = Particle(p2_four_vector_LF, self.get_rf(), p2_dict)
        new_particle_3 = Particle(p3_four_vector_LF, self.get_rf(), p3_dict)
        
        return [new_particle_1, new_particle_2, new_particle_3]

    def decay_particle(self):
        if self.get_ids()["PID"] not in meson_decay_dict.keys():
            raise ValueError("Decay options for particle not specified. Edit dictionary in 'particle.py' to include it")
        decay_options = meson_decay_dict[self.get_ids()["PID"]]
        if len(decay_options) == 1:
            br_sum, decay = decay_options[0]
        else:
            branching_ratios = [decay_options[ii][0] for ii in range(len(decay_options))]
            br_sum = np.sum(branching_ratios)
            choice_weights = branching_ratios/br_sum
            decay = decay_options[np.random.choice(range(len(decay_options)), p=choice_weights)][1]

        if len(decay) > 2:
            raise ValueError("Three-body (and above) decays not yet implemented")
        elif len(decay) == 2:
            p1_dict = {"PID":decay[0], "weight":self.get_ids()["weight"]*br_sum, "ID":2*(self.get_ids()["ID"]), "generation_process":"SMDecay", "generation_number":(self.get_ids()["generation_number"]+1), "production_time":self.get_ids()["decay_time"]}
            p2_dict = {"PID":decay[1], "weight":self.get_ids()["weight"]*br_sum, "ID":2*(self.get_ids()["ID"])+1, "generation_process":"SMDecay", "generation_number":(self.get_ids()["generation_number"]+1), "production_time":self.get_ids()["decay_time"]}
            new_particles = self.two_body_decay(p1_dict=p1_dict, p2_dict=p2_dict)
        
        self.set_ended(True)
        return new_particles        