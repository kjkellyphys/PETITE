#!/usr/bin/env python
import pathlib
from setuptools import setup

# Parent directory
HERE = pathlib.Path(__file__).parent

# The readme file
README = (HERE / "README.md").read_text()

setup(
    name='PETITE',
    package_dir={'PETITE':'src'},
    version="1.0.0",
    author ="Nikita Blinov, Patrick J. Fox, Kevin J. Kelly, Pedro A.N. Machado, Ryan Plestid",
    author_email="nblinov@yorku.ca, pjfox@fnal.gov, kjkelly@tamu.edu, pmachado@fnal.gov, rplestid@caltech.edu",
    description="Package for Electromagnetic Transitions in Thick-target Environments\
                 Allows for simulations of standard model electromagnetic showers and\
                 production of new-physics particles in beam-dump environments.",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/kjkellyphys/PETITE",
    install_requires=[
        "vegas>=5.4.2",
        "tqdm",
        "numpy"#==1.24"
    ],
    extras_require={
        "interactive": ["nbstripout", "matplotlib", "jupyter"]
    },
    packages=["PETITE"],
)
