from setuptools import setup, find_packages

setup(
    name='tap_magnitudes',
    version='1.0',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    install_requires=[
        'pandas',
        'numpy',
        'scipy'
    ],
    include_package_data=True,
    package_data={'tap_magnitudes':['transmission_curves/**/*']}
)