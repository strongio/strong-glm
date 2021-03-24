from setuptools import setup, find_packages
from strong_glm._version import __version__

setup(
    name='strong-glm',
    version=__version__,
    url='http://github.com/strongio/strong-glm',
    author='Jacob Dink',
    author_email='jacob.dink@strong.io',
    packages=find_packages(),
    install_requires=[
        'numpy>=1.17.2',
        "scipy>=1.3.1",
        "skorch>=0.9.0",
        "torch>=1.6.0"
    ]
)
