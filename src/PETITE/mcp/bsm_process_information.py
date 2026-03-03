from PETITE.mcp.bsm_processes import *
from PETITE.mcp.bsm_kinematics import *

#-------------------------------------------------
# List of parameters required to specify the model
#-------------------------------------------------
model_required_keys = {"m_chi",      #Millicharged Particle Mass
                       }

default_key_values = {"m_chi":0.010} #
organizational_key = "m_chi" # Label required for organizing pre-computed integrator maps
                          # Should be a mass in GeV.

bsm_process_info = {}
bsm_process_counter = 0

bsm_process_info['mcpAnnihilation'] = {}
bsm_process_info['mcpAnnihilation']['dimensionalities'] = 1
bsm_process_info['mcpAnnihilation']['scattering_target'] = 'electron'
bsm_process_info['mcpAnnihilation']['incoming_particle_PID'] = -11
bsm_process_info['mcpAnnihilation']['outgoing_particle_PID'] = np.array([52, -52]) # outgoing millicharged particle pair
bsm_process_info['mcpAnnihilation']['differential_cross_section'] = dsigma_mcp_annihilation_dcostheta
bsm_process_info['mcpAnnihilation']['kinematic_function'] = mcp_annihilation_fourvecs
bsm_process_info['mcpAnnihilation']['process_code'] = bsm_process_counter
bsm_process_counter += 1