import setuptools

setuptools.setup(
    name='PETITE',
    package_dir={'PETITE':'src'},
    #package_dir={'PETITE':'data','PETITE':'src'},
    #include_package_data=True,
    #data_files=[('PETITE', ['data/beams/lumi_integral_list.dat'])],
    version="0.0.0",
    author ="Kevin J. Kelly",
    description="Packages...",
    packages=["PETITE"]
)
