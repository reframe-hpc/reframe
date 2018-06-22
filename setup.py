import setuptools

from reframe import VERSION

with open('README.md') as read_me:
    long_description = ''.join(read_me.readlines()[2:])

setuptools.setup(
    name='ReFrame-HPC',
    version=VERSION,
    author='CSCS Swiss National Supercomputing Center',
    description='ReFrame is a new framework for writing regression tests '
                'for HPC systems',
    long_description=long_description,
    long_description_content_type='ext/markdown',
    url='https://github.com/eth-cscs/reframe',
    licence='BSD 3-Clause',
    packages=setuptools.find_packages(exclude=['unittests']),
    python_requires='>=3.5',
    scripts=['bin/reframe'],
    classifiers=(
        'Developement Status :: 5 - Production/Stable',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'License :: OSI Approved :: BSD Licence',
        'Operating System :: MacOS',
        'Operating System :: POSIX :: Linux',
        'Environment :: Console'
    ),
)
