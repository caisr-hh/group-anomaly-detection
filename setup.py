from setuptools import setup

setup(name='grand',
    version='0.1',
    description='GRAND: Group-based Anomaly Detection for Large-Scale Monitoring of Complex Systems',
    url='https://github.com/caisr-hh',
    author='Mohamed-Rafik Bouguelia - Center for Applied Intelligent Systems Research (CAISR)',
    author_email='mohbou@hh.se',
    license='MIT',
    packages=['grand'],
    install_requires=['matplotlib>=2.1.0', 'numpy>=1.13.3', 'pandas>=0.22.0', 'scipy>=1.0.0', 'scikit-learn>=0.20.0'],
    zip_safe=False)
    
