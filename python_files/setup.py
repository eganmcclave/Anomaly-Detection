from distutils.core import setup
from Cython.Build import cythonize

setup(
    name='Anomaly-Speed (Python Version)',
    version='0.3',
    description='Anomaly Speed Detection Project for 36-650',
    packages=['code'],
    install_requires=['numpy']
)
