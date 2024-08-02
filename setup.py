from setuptools import setup, find_packages

def open_requirements(path):
    with open(path) as f:
        requires = [
            r.split('/')[-1] if r.startswith('git+') else r
            for r in f.read().splitlines()]
    return requires

requires = open_requirements('requirements.txt')
setup(
    name='pylars-sipm',
    version='0.3.0',
    packages=find_packages(exclude=['tests*']),
    license='none',
    description='A python-based simple processor for data acquired with XenoDAQ',
    long_description=open('README.md').read(),
    install_requires=requires,
    url='https://github.com/ricmperes/PyLArS',
    author='Ricardo Peres',
    author_email='rperes@physik.uzh.ch',
    scripts=['bin/pylars', 'bin/pylars_led', 'bin/pylars_gainevo']
)