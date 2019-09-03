import setuptools

from reframe import VERSION

with open('README.md') as read_me:
    long_description = ''.join(read_me.readlines()[6:])

setuptools.setup(
    name='ReFrame-HPC',
    version=VERSION,
    author='CSCS Swiss National Supercomputing Center',
    description='ReFrame is a framework for writing regression tests '
                'for HPC systems',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/eth-cscs/reframe',
    licence='BSD 3-Clause',
    packages=setuptools.find_packages(exclude=['unittests']),
    python_requires='>=3.5',
    scripts=['bin/reframe'],
    classifiers=(
        'Development Status :: 5 - Production/Stable',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'License :: OSI Approved :: BSD License',
        'Operating System :: MacOS',
        'Operating System :: POSIX :: Linux',
        'Environment :: Console'
    ),
)
