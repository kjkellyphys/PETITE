from PETITE.sm.sm_processes import *
from PETITE.sm.kinematics import *

sm_process_info = {}
sm_process_counter = 0

sm_process_info['Brem'] = {}
sm_process_info['Brem']['dimensionalities'] = 4 #number of outgoing phase space variables
sm_process_info['Brem']['scattering_target'] = 'nucleus'
#sm_process_info['Brem']['form_factor_function'] = g2_elastic
#sm_process_info['Brem']['q_sq_function'] = brem_q_sq_dimensionless
sm_process_info['Brem']['incoming_particle_PID'] = np.array([11, -11]) #electron or positron
sm_process_info['Brem']['outgoing_particle_PID'] = np.array([0, 22]) #outgoing particle of interest: photon
                                                                     #0 indicates that incoming particle is unchanged
sm_process_info['Brem']['differential_cross_section'] = dsigma_brem_dimensionless
sm_process_info['Brem']['kinematic_function'] = l_to_lgamma_fourvecs
sm_process_info['Brem']['process_code'] = sm_process_counter
sm_process_counter += 1

sm_process_info['PairProd'] = {}
sm_process_info['PairProd']['dimensionalities'] = 4
sm_process_info['PairProd']['scattering_target'] = 'nucleus'
#sm_process_info['PairProd']['form_factor_function'] = g2_elastic
#sm_process_info['PairProd']['q_sq_function'] = pair_production_q_sq_dimensionless
sm_process_info['PairProd']['incoming_particle_PID'] = 22
sm_process_info['PairProd']['outgoing_particle_PID'] = np.array([-11, 11]) #e+ e-
sm_process_info['PairProd']['differential_cross_section'] = dsigma_pairprod_dimensionless
sm_process_info['PairProd']['kinematic_function'] = gamma_to_epem_fourvecs
sm_process_info['PairProd']['process_code'] = sm_process_counter
sm_process_counter += 1

#########################################
# Compton Scattering gamma e- -> gamma e-
#########################################
sm_process_info['Comp'] = {}
sm_process_info['Comp']['dimensionalities'] = 1
sm_process_info['Comp']['scattering_target'] = 'electron'
#sm_process_info['Comp']['form_factor_function'] = unity
#sm_process_info['Comp']['q_sq_function'] = dummy
sm_process_info['Comp']['incoming_particle_PID'] = 22
sm_process_info['Comp']['outgoing_particle_PID'] = np.array([22, 11]) #photon and electron
sm_process_info['Comp']['differential_cross_section'] = dsigma_compton_dCT
sm_process_info['Comp']['kinematic_function'] = compton_fourvecs
sm_process_info['Comp']['process_code'] = sm_process_counter
sm_process_counter += 1

#########################################
# Annihilation e+ e- -> gamma gamma
#########################################
sm_process_info['Ann'] = {}
sm_process_info['Ann']['dimensionalities'] = 1
sm_process_info['Ann']['scattering_target'] = 'electron'
#sm_process_info['Ann']['form_factor_function'] = unity
#sm_process_info['Ann']['q_sq_function'] = dummy
sm_process_info['Ann']['incoming_particle_PID'] = -11
sm_process_info['Ann']['outgoing_particle_PID'] = np.array([22, 22]) #two photons
sm_process_info['Ann']['differential_cross_section'] = dsigma_annihilation_dCT
sm_process_info['Ann']['kinematic_function'] = annihilation_fourvecs
sm_process_info['Ann']['process_code'] = sm_process_counter
sm_process_counter += 1

#########################################
# Moller Scattering e- e- -> e- e-
#########################################
sm_process_info['Moller'] = {}
sm_process_info['Moller']['dimensionalities'] = 1
sm_process_info['Moller']['scattering_target'] = 'electron'
#sm_process_info['Moller']['form_factor_function'] = unity
#sm_process_info['Moller']['q_sq_function'] = dummy
sm_process_info['Moller']['incoming_particle_PID'] = 11
sm_process_info['Moller']['outgoing_particle_PID'] = np.array([11, 11]) #two electrons
sm_process_info['Moller']['differential_cross_section'] = dsigma_moller_dCT
sm_process_info['Moller']['kinematic_function'] = ee_to_ee_fourvecs
sm_process_info['Moller']['process_code'] = sm_process_counter
sm_process_counter += 1

#########################################
# Bhabha Scattering e+ e- -> e+ e-
#########################################
sm_process_info['Bhabha'] = {}
sm_process_info['Bhabha']['dimensionalities'] = 1
sm_process_info['Bhabha']['scattering_target'] = 'electron'
#sm_process_info['Bhabha']['form_factor_function'] = unity
#sm_process_info['Bhabha']['q_sq_function'] = dummy
sm_process_info['Bhabha']['incoming_particle_PID'] = -11
sm_process_info['Bhabha']['outgoing_particle_PID'] = np.array([-11, 11]) #positron and electron
sm_process_info['Bhabha']['differential_cross_section'] = dsigma_bhabha_dCT
sm_process_info['Bhabha']['kinematic_function'] = ee_to_ee_fourvecs
sm_process_info['Bhabha']['process_code'] = sm_process_counter
sm_process_counter += 1

##########################################
# Muon Bremsstrahlung mu- N -> mu- N gamma
##########################################
sm_process_info['MuonBrem'] = {}
sm_process_info['MuonBrem']['dimensionalities'] = 4
sm_process_info['MuonBrem']['scattering_target'] = 'nucleus'
#sm_process_info['MuonBrem']['form_factor_function'] = g2_elastic
#sm_process_info['MuonBrem']['q_sq_function'] = muon_brem_q_sq_dimensionless
sm_process_info['MuonBrem']['incoming_particle_PID'] = np.array([13, -13]) #muon or anti-muon
sm_process_info['MuonBrem']['outgoing_particle_PID'] = np.array([0, 22]) #outgoing particle of interest: photon
                                                                     #0 indicates that incoming particle is unchanged
sm_process_info['MuonBrem']['differential_cross_section'] = dsigma_brem_dimensionless
sm_process_info['MuonBrem']['kinematic_function'] = l_to_lgamma_fourvecs
sm_process_info['MuonBrem']['process_code'] = sm_process_counter
sm_process_counter += 1

###########################################
# Muon-Electron Scattering mu- e- -> mu- e-
###########################################
sm_process_info['MuonE'] = {}
sm_process_info['MuonE']['dimensionalities'] = 1
sm_process_info['MuonE']['scattering_target'] = 'electron'
#sm_process_info['MuonE']['form_factor_function'] = unity
#sm_process_info['MuonE']['q_sq_function'] = dummy
sm_process_info['MuonE']['incoming_particle_PID'] = np.array([13, -13]) #muon or anti-muon
sm_process_info['MuonE']['outgoing_particle_PID'] = np.array([0, -11]) #muon/antimuon and electron
sm_process_info['MuonE']['differential_cross_section'] = dsigma_muonelectron_dCT
sm_process_info['MuonE']['kinematic_function'] = mue_to_mue_fourvecs
sm_process_info['MuonE']['process_code'] = sm_process_counter
sm_process_counter += 1