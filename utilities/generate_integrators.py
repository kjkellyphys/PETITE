""" Create utility that makes readme explaining the pickles in a specific directory
    Pickles directory can be an input in command line -pickle_dir=/blahblah/ (target name)
    SM ==> one run
    DARK ==> one per mass and target (only dark brem is target-specific)
    Run find_maxes after generating pickles to generate dicts (once per material)
    Save readme in Ryan dicts and directory
    NBP ==> user-defined, but we will have ours: raw_integrators
    Ryan dicts ==> cooked_integrators
    FIXME: units and constants
    FIXME: manual
    FIXME: comments
"""

import numpy as np
# import sys
#sys.path.insert(0,'../PETITE/')
# sys.path.insert(0,'/Users/pmachado/Dropbox/Projects/main/dark showers/feb/PETITE-main/PETITE')
import sys, os
path = os.getcwd()
path = os.path.join(path,"../PETITE")
sys.path.insert(0,path)
from AllProcesses import *
from physical_constants import *

import os
from copy import deepcopy
import random as rnd
import itertools
from multiprocessing import Pool
import pickle
from functools import partial
import argparse

def create_param_dict(mV, A, Z, mT):
    """
    Creates a parameter dictionary
    """
    params = {'me' : 511e-6, 'alphaEM': 1./137}
    params.update({'A': A, 'Z': Z, 'mT': mT})
    params.update({'mV':mV})
    return params


def generate_vector_mass_string(mV):
    return str(int(np.floor(mV*1000.)))+"MeV"
    

#file management for SM processes
def run_vegas_in_parallel(params, process, verbosity_mode, file_info, energy_index):
    params['E_inc'] = params['initial_energy_list'][energy_index]
    [brem_pickle_dir, brem_pickle_temp_dir] = file_info
    strsaveB = brem_pickle_dir + str(energy_index) + ".p" # FIXED
    if os.path.exists(strsaveB):
        print("Already generated integrator for this point\n")
    else:
        print('Starting VEGAS for energy index ',energy_index)
        VEGAS_integrator = VEGASIntegration(params, process, VB=verbosity_mode, mode='Pickle') # FIXED
        print('Done VEGAS for energy index ',energy_index)
        pickle.dump(VEGAS_integrator, open(strsaveB, "wb"))
        print('File created: '+strsaveB)
    return()



def make_integrators(params, process, verbosity_mode):
    """
    Generate vegas integrator pickles for the following parameters:
        mV : dark vector mass in GeV
        A : target atomic mass number
        Z : target atomic number
        mT : target mass in GeV
    """
    mV = params['mV']
    A = params['A']
    Z = params['Z']
    mT = params['mT'] 
    params['m_e'] = m_electron#511E-6 #electron mass in GeV PJF--FIX THIS ??
    params['alpha_FS'] = alpha_em#1.0/137 #PJF--FIX THIS ??

    initial_energy_list = params['initial_energy_list']
    target_name = params['target_name']
    
    vec_mass_string = generate_vector_mass_string(mV)
    energy_index_list = range(len(initial_energy_list))

    print("Parameters:")
    print(params)
    print('Doing process: ', process)
    
    # energy_index_list = range(len(initial_energy_list))
    # vec_mass_string = generate_vector_mass_string(params['mV'])

    brem_pickle_dir = '../NBP/'+params['target_name']+'/DarkV/'
    brem_pickle_temp_dir = brem_pickle_dir + "DarkBremPickes_TMP/"

    file_info = [brem_pickle_dir, brem_pickle_temp_dir]

    if os.path.exists(brem_pickle_temp_dir) == False:
        os.system("mkdir -p " + brem_pickle_temp_dir)
    # PJF -- make check if file already exists...to save time!!
    brem_file_prefix = brem_pickle_temp_dir + "BremPickles_" + vec_mass_string + "_"
    pool = Pool()
    res = pool.map(partial(run_vegas_in_parallel, params, process, verbosity_mode, file_info), energy_index_list)
    print('make_integrators - done')



    return()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Produce VEGAS integrators for various production processes')    
    # mandatory parameters
    
    parser.add_argument('-A', type=float, help='atomic mass number', required=True)
    parser.add_argument('-Z', type=float, help='atomic number', required=True)
    parser.add_argument('-mT', type=float, help='nuclear target mass in GeV', required=True)
    parser.add_argument('-target_name', type=str, help='name of the target', required=True)

    # optional parameters
    parser.add_argument('-process', nargs='+', type=str, default='DarkBrem', help='list of processes to be run "all" does whole list, if mV non-zero only DarkBrem \
        (choose from "PairProd", "Brem", "DarkBrem", "Comp", "Ann")')
    parser.add_argument('-mV', nargs='+', type=float, default=[0.05], help='dark vector mass in GeV (can be a space-separated list)')
    parser.add_argument('-num_energy_pts', type=int, default=100, help='number of initial energy values to evaluate')
    parser.add_argument('-min_energy', type=float, default=0.01, help='minimum initial energy to evaluate (must be larger than mV)')
    parser.add_argument('-max_energy', type=float, default=100., help='maximum initial energy to evaluate')
    parser.add_argument('-verbosity', type=bool, default=False, help='verbosity mode True/False')

    args = parser.parse_args()


    params = {'A': args.A, 'Z': args.Z, 'mT': args.mT, 'target_name': args.target_name}
    verbosity_mode = args.verbosity
    if (args.mV == 0):# doing SM processes
        if  "all" in args.process:
            process_list_to_do = ['Brem','PairProd','Comp','Ann']
        else:#make sure DarkBrem not accidentally in list
            try:
                process_list_to_do = args.process.remove('DarkBrem')
            except:
                pass
        for process in args.process:
            initial_energy_list = np.logspace(np.log10(args.min_energy), np.log10(args.max_energy), args.num_energy_pts)
            params.update({'mV' : 0})
            params.update({'initial_energy_list': initial_energy_list})
            make_integrators(params, process, verbosity_mode)
    else:# doing DarkBrem
        for mV in args.mV:
            process = 'DarkBrem'
            print("Working on mV = ", mV)
            min_energy = min(args.min_energy, 1.01 * mV)
            initial_energy_list = np.logspace(np.log10(min_energy), np.log10(args.max_energy), args.num_energy_pts)
            params.update({'mV' : mV})
            params.update({'initial_energy_list': initial_energy_list})
            make_integrators(params, process, verbosity_mode)




    

        
