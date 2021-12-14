from setuptools import setup, find_packages

setup(
    name='SBIExperiments',
    version='0.0.1',
    url='https://github.com/astrodeepnet/sbi_experiments',
    author='Justine Zeghal and friends',
    description='Package for numerical experiments of SBI tools',
    packages=find_packages(),
    install_requires=[
        'numpy>=1.19.2',
        'jax>=0.2.0',
        'jaxlib>=0.1.74',
        'tensorflow_probability>=0.14.1',
        'scikit-learn>=0.21'
    ],
)
