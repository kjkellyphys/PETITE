# PETITE
Monte Carlo generator for production of dark sector objects in thick-target experiments
PETITE generates a standard model electromagnetic shower due to an electron, positron or photon propagating through a dense medium.

## Installation
To install, from the top directory run

    >> pip install .

## Running PETITE
TBD

We first describe how to run PETITE using the pre-generated VEGAS integrator files.  It is also possible to generate new files for both SM and dark processes.  The tools necessary for this are in the utilities directory and we describe their use below, as well as in utilities/README.md.  Jupyter notebook examples of how to run PETITE using pre-generated files are given in the examples directory.

First one must import the VEGAS integrators and other infrastructure for the physicsal processes in the relevant material.  This is done using Shower and DarkShower.  With these loaded one may then generate a full shower by calling generate_(dark_)shower, or investigate individual physical processes using draw_(dark_)sample.

Reorder tutorial to have running order -- event display, validation, then individual processes/messign with VEGAS.  
FIXME: Why does shower not work if run twice in a row??



# Generating a full SM shower:
    - Define initial particle that seeds shower eg an electron with 4-momentum [E0,px,py,pz] entering at position [0,0,0] `Particle([E0,px,py,pz], [0,0,0], {"PID":11})`
    - Setup the necessary infrastructure for a shower sGraphite = Shower(<directory containing dictionaries>, <material>, <min energy down to which particles are trcked, in GeV>) eg 
    `sGraphite = Shower(dictionary_dir_TEMP, "graphite", 0.010)`
    - Generate a shower `standard_shower = sGraphite.generate_shower(p0, VB=True)`

    






# Pre-generated VEGAS integrators
PETITE comes with pre-generated VEGAS integrators (both before and after processing) for standard showers in graphite and lead.
You can find them in the ./data/

## Structure of pre-generated VEGAS adaptive maps
In ./data/VEGAS_backend one can find a list of adaptive maps for both standard and dark showers, for each process.
#FIXME: were they done for hydrogen?

In ./data/VEGAS_dictionaries you have processed versions of the adaptive maps, with the following structure:

### Pre-generated processed standard model showers
sm_maps.pkl is a dictionary of dictionaries.
The first layer of keys are the standard model processes:
    'PairProd', 'Comp', 'Ann', 'Brem', 'Moller' and 'Bhabha'
For each process we there are several entries corresponding to different incoming energy.
Each entry is a vector in which the first component is the incoming energy, while the second has a dictionary with VEGAS parameters and the adaptive map ('neval', 'max_F', 'Eg_min', 'adaptive_maps').

Besides the adaptive maps, there are also cross section tables for each process, which are used to determine the process that occurs in each step of the shower evolution. These can be found in sm_xsecs.pkl.  The pre-generated file is a dictionary where the first two keys are the same standard model processes as above, followed by the material the beam is assumed to be transiting,  graphite and lead, followed by an array of a (particle energy, cross section).

### Pre-generated processed dark showers
dark_maps.pkl is a dictionary of dictionaries.
The first layer of keys are the masses of the dark vector.
For each mass, the second layer of keys are the dark shower processes:
    'Brem', 'Comp', and 'Ann'
Inside those, everything follows the same structure as for standard showers.

Cross sections for dark showers can be found in dark_xsecs.pkl.  Just as for sm_maps relates to dark_maps, sm_xsecs relates to dark_xsecs, there is one additional initial layer of keys labelling the dark photon mass.

# TODO
- [ ] Explain how to run PETITE using pre-generated pickles
- [ ] Explain how to generate pickles for given target
- [ ] Explain how to generate pickles for given target and dark sector model





# OLD (to be removed)
Run the script "Gen_Samples.py" to make unweighted samples for lead and graphite simulations.

The main code is in "shower.py" for standard model showers and "dark_shower.py", and an example script for using this code is "Example_GeVElectronShower.py".

