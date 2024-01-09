import setuptools

setuptools.setup(
    name='PETITE',
    package_dir={'PETITE':'src'},
    version="1.0.0",
    author ="Nikita Blinov, Patrick J. Fox, Kevin J. Kelly, Pedro A.N. Machado, Ryan Plestid",
    description="Package for Electromagnetic Transitions in Thick-target Environments\
                 Allows for simulations of standard model electromagnetic showers and\
                 production of new-physics particles in beam-dump environments.",
    packages=["PETITE"]
)
