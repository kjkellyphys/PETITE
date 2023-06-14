# PETITE
Monte Carlo generator for production of dark sector objects in thick-target experiments
PETITE generates a standard model electromagnetic shower due to an electron, positron or photon propagating through a dense medium.

## Installation
To install, from the top directory run

    >> pip install .

PETITE comes with pre-generated showers for graphite and lead #FIXME: energies?

## Running PETITE
TBD

# Pre-generated VEGAS integrators
PETITE comes with pre-generated VEGAS integrators (both before and after processing) for standard showers in graphite and lead.
You can find them in the ./data/

## Structure of pre-generated VEGAS adaptive maps
In ./data/VEGAS_backend one can find a list of adaptive maps for both standard and dark showers, for each process.
#FIXME: were they done for hydrogen?

In ./data/VEGAS_dictionaries you have processed versions of the adaptive maps, with the following structure:

### Pre-generated processed standard model showers
standard_maps.pkl is a dictionary of dictionaries.
The first layer of keys are the standard model processes:
    'PairProd', 'Comp', 'Ann', 'Brem', 'Moller' and 'Bhabha'
For each process we there are several entries corresponding to different incoming energy.
Each entry is a vector in which the first component is the incoming energy, while the second has a dictionary with VEGAS parameters and the adaptive map ('neval', 'max_F', 'Eg_min', 'adaptive_maps').

Besides the adaptive maps

### Pre-generated processed dark showers
dark_maps.pkl is a dictionary of dictionaries.
The first layer of keys are the masses of the dark vector.
For each mass, the second layer of keys are the dark shower processes:
    'Brem', 'Comp', and 'Ann'
Inside those, everything follows the same structure as for standard showers.


# TODO
- [ ] Explain how to run PETITE using pre-generated pickles
- [ ] Explain how to generate pickles for given target
- [ ] Explain how to generate pickles for given target and dark sector model





# OLD (to be removed)
Run the script "Gen_Samples.py" to make unweighted samples for lead and graphite simulations.

The main code is in "shower.py" for standard model showers and "dark_shower.py", and an example script for using this code is "Example_GeVElectronShower.py".

