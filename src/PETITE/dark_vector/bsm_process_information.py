from PETITE.dark_vector.bsm_processes import *
from PETITE.dark_vector.bsm_kinematics import *
from PETITE.sm import sm_processes as smpr
from PETITE.sm import kinematics as smkin

bsm_process_info = {}
bsm_process_counter = 0

bsm_process_info['DarkBrem'] = {}
bsm_process_info['DarkBrem']['dimensionalities'] = 3
bsm_process_info['DarkBrem']['scattering_target'] = 'nucleus'
bsm_process_info['DarkBrem']['incoming_particle_PID'] = np.array([11, -11]) #electron or positron
bsm_process_info['DarkBrem']['outgoing_particle_PID'] = np.array([0, 4900022]) #outgoing particle of interest: dark photon
                                                                               #0 indicates that incoming particle is unchanged    
                                                                               #4900022 is dark photon PDG code
bsm_process_info['DarkBrem']['differential_cross_section'] = dsig_dx_dcostheta_dark_brem_exact_tree_level
bsm_process_info['DarkBrem']['kinematic_function'] = l_to_lV_fourvecs
bsm_process_info['DarkBrem']['process_code'] = bsm_process_counter
bsm_process_counter += 1

bsm_process_info['DarkComp'] = {}
bsm_process_info['DarkComp']['dimensionalities'] = 1
bsm_process_info['DarkComp']['scattering_target'] = 'electron'
bsm_process_info['DarkComp']['incoming_particle_PID'] = 22
bsm_process_info['DarkComp']['outgoing_particle_PID'] = np.array([11, 4900022]) # electron and dark photon
bsm_process_info['DarkComp']['differential_cross_section'] = smpr.dsigma_compton_dCT
bsm_process_info['DarkComp']['kinematic_function'] = compton_fourvecs
bsm_process_info['DarkComp']['process_code'] = bsm_process_counter
bsm_process_counter += 1

bsm_process_info['DarkAnn'] = {}
bsm_process_info['DarkAnn']['dimensionalities'] = 1
bsm_process_info['DarkAnn']['scattering_target'] = 'electron'
bsm_process_info['DarkAnn']['incoming_particle_PID'] = -11
bsm_process_info['DarkAnn']['outgoing_particle_PID'] = np.array([22, 4900022]) # photon and dark photon
bsm_process_info['DarkAnn']['differential_cross_section'] = dsigma_radiative_return_du
bsm_process_info['DarkAnn']['kinematic_function'] = annihilation_fourvecs
bsm_process_info['DarkAnn']['process_code'] = bsm_process_counter
bsm_process_counter += 1

bsm_process_info['DarkMuonBrem'] = {}
bsm_process_info['DarkMuonBrem']['dimensionalities'] = 3
bsm_process_info['DarkMuonBrem']['scattering_target'] = 'nucleus'
bsm_process_info['DarkMuonBrem']['incoming_particle_PID'] = np.array([13, -13]) #muon or anti-muon
bsm_process_info['DarkMuonBrem']['outgoing_particle_PID'] = np.array([0, 4900022]) #outgoing particle of interest: dark photon
                                                                               #0 indicates that incoming particle is unchanged    
                                                                               #4900022 is dark photon PDG code
bsm_process_info['DarkMuonBrem']['differential_cross_section'] = dsig_dx_dcostheta_dark_brem_exact_tree_level
bsm_process_info['DarkMuonBrem']['kinematic_function'] = l_to_lV_fourvecs
bsm_process_info['DarkMuonBrem']['process_code'] = bsm_process_counter
bsm_process_counter += 1
