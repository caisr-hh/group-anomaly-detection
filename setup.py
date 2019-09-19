from setuptools import setup, find_packages
import os, re

verfile = ".{}grand{}_version.py".format(os.sep, os.sep)
verstrline = open(verfile, "rt").read()
VSRE = r"^__version__ = ['\"]([^'\"]*)['\"]"
mo = re.search(VSRE, verstrline, re.M)
version_str = mo.group(1)

data_path = '.{}grand{}datasets{}data'.format(os.sep, os.sep, os.sep)
data_files = [ 'datasets' + os.sep + 'data' + os.sep + dirname + os.sep + '*' for dirname in os.listdir(data_path) ]

setup(name='grand',
    version=version_str,
    description='GRAND: Group-based Anomaly Detection for Large-Scale Monitoring of Complex Systems',
    url='https://github.com/caisr-hh',
    author='Mohamed-Rafik Bouguelia - Center for Applied Intelligent Systems Research (CAISR)',
    author_email='mohbou@hh.se',
    license='MIT',
    packages=find_packages(),
    package_data={'grand':data_files},
    install_requires=['matplotlib>=2.1.0', 'numpy>=1.13.3', 'pandas>=0.22.0', 'scipy>=1.0.0', 'scikit-learn>=0.20.0'],
    zip_safe=False,
    python_requires='>=3.4')
