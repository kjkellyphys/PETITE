# PETITE
PETITE: Package for Electromagnetic Transitions In Thick-target Environments
Monte Carlo generator for production of dark sector objects in thick-target experiments
PETITE generates electromagnetic showers for incoming electron, positron or photon propagating through a dense medium, and includes the possibility of dark sector particle production.

## Installation
To install, from the top directory run
 > pip install .

### Dependencies
PETITE, its tutorials and tools require the following packages: numpy 1.24, vegas (>= 5.4.2), cProfile, pickle, matplotlib, scipy, datetime, tqdm, copy, sys, random and functools. Using `pip install .` should install all requirements, but if needed, you can manually install these packages with
 > pip install <package_name>==<version_required>

## Running PETITE
*The notebook `./examples/tutorial.ipynb` offers an alternative to this readme.*

PETITE comes with pre-generated VEGAS integrator files stored in the `./data/VEGAS_dictionaries/` directory. 
Key functions (in `./src/shower.py` and `./src/dark_shower.py`):
- `Shower` and `DarkShower` classes: import the VEGAS integrators and other infrastructure for the physical processes in the relevant material.
- `generate_shower` and `generate_dark_shower` functions: generate a shower for a given initial `Particle`.

### Generating a full SM shower
(1) Define initial particle that seeds shower e.g. an electron with 4-momentum [E0,px,py,pz] entering at position [0,0,0]:
 > incoming_electron = Particle([E0,px,py,pz], [0,0,0], {"PID":11})`

(2) Setup the necessary infrastructure for a shower with `sGraphite = Shower(<directory containing dictionaries>, <material>, <min energy down to which particles are tracked, in GeV>)`, e.g.
 > sGraphite = Shower("./data/", "graphite", 0.010)

(3) Generate the shower 
 > standard_shower = sGraphite.generate_shower(incoming_electron, VB=True)

The output of `generate_shower` is a list of `Particle` objects generated through the development of the shower.

### Generating a full dark shower
(1) As for the standard shower, define initial particle that seeds shower

(2) Setup the necessary infrastructure for a shower with `sGraphite = DarkShower(<directory containing dictionaries>, <material>, <min energy down to which particles are tracked, in GeV>, <dark particle mass, in GeV>)`, e.g.
 > sGraphite = DarkShower("./data/", "graphite", 0.010, 0.001)

(3) Generate the shower 
 > dark_shower = sGraphite.generate_dark_shower(incoming_electron, VB=True)

The output of `generate_dark_shower` is a list of `Particle` objects generated through the development of the shower, which includes dark vectors.

We can plot event displays for both standard and dark shower with 
 > event_display(shower_object)

FIXME: Add how it translates to a number of events in a detector

### The `Particle` object
Particles are stored in PETITE as `Particle` objects, (see `./src/particle.py`). The `Particle` class has the following attributes:
- `p0`: 4-momentum of the particle at the start of the shower. If given a scalar, it is assumed to be the energy of a particle propagating in the z-axis.
- `r0`: 3-position of the particle at the start of the shower. 
- `id_dictionary`: dictionary of particle properties, including
    - `PID`: particle ID number
    - `ID`: ID for shower development (unique for each particle)
    - `parent_ID`: shower ID of parent particle
    - `generation number`: number of generations from initial particle
    - `mass`: particle mass
    - `generation_process`: process that generated the particle
    - `weight`: weight of the particle, used only for dark showers
    - `stability`: whether the particle is stable or not (PETITE can perform isotropic 2-body decays of unstable particles such as pi0s)

These attributes can be used for analyzing the shower, see `./examples/tutorial.ipynb` for examples.

*This concludes the minimum information needed to run PETITE.  The following sections are intended for advanced users.*

# Advanced usage
The following is intended for advanced users who wish to generate their own VEGAS integrators for a given material and/or dark sector model, to understand better the VEGAS sampling procedure, or to understand the file structure of the integrators.

## Generating new VEGAS integrators
It is possible to generate new files for both SM and dark processes using tools in the `./utilities/` directory and we describe their use below, as well as in `./utilities/README.md`.  Jupyter notebook examples of how to run PETITE using pre-generated files are given in the `./examples/` directory.

## Structure of pre-generated VEGAS adaptive maps
In `./data/auxiliary/` one can find a list of adaptive maps for both standard and dark showers, for each process.
In `./data/` you have processed versions of the adaptive maps, with the structure as decribed below.

### Pre-generated processed standard model showers
`sm_maps.pkl` is a dictionary of dictionaries.
The first layer of keys are the standard model processes:
    `PairProd`, `Comp`, `Ann`, `Brem`, `Moller` and `Bhabha`.
For each process we there are several entries corresponding to different incoming energy.
Each entry is a vector in which the first component is the incoming energy, while the second has a dictionary with VEGAS parameters and the adaptive map `{'neval', 'max_F', 'Eg_min', 'adaptive_maps'}`.

Besides the adaptive maps, there are also cross section tables for each process, which are used to determine the process that occurs in each step of the shower evolution. These can be found in `sm_xsecs.pkl`.  The pre-generated file is a dictionary where the first two keys are the same standard model processes as above, followed by the material the beam is assumed to be transiting,  graphite and lead, followed by an array of a (particle energy, cross section).

### Pre-generated processed dark showers
`dark_maps.pkl` is also a dictionary of dictionaries.
The first layer of keys are the masses of the dark vector in GeV (e.g. 0.1).
For each mass, the second layer of keys are the dark shower processes:
    `DarkBrem`, 'DarkComp', and 'DarkAnn'
Inside those, everything follows the same structure as for standard showers.

Cross sections for dark showers can be found in `dark_xsecs.pkl`.  Just as for `sm_maps` relates to `dark_maps`, `sm_xsecs` relates to `dark_xsecs`, there is one additional initial layer of keys labelling the dark particle mass.