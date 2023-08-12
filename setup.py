import os.path
from setuptools import setup, find_packages


def __read_requirements():
    with open(os.path.join(os.path.split(__file__)[0], 'requirements.txt'), 'r') as f:
        return [s.strip() for s in f.readlines()]


setup(
    name='sawdown',
    version='0.1',
    packages=find_packages(exclude=['tests']),
    description='Optimization tools',
    long_description=open('README.md', 'rb').read().decode('utf8'),
    author='Thor Hendricks',
    install_requires=__read_requirements(),
)

