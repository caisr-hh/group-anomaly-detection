from setuptools import setup

setup(name='cosmo',
    version='0.1',
    description='Self-Monitoring with Group-based Anomaly Detection',
    url='https://github.com/caisr-hh',
    author='CAISR - Center for Applied Intelligent Systems Research',
    author_email='mohbou@hh.se',
    license='MIT',
    packages=['cosmo'],
    install_requires=['matplotlib>=2.1.0', 'numpy>=1.13.3', 'pandas>=0.22.0', 'scipy>=1.0.0'],
    zip_safe=False)
    