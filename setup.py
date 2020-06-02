# Copyright 2016-2020 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import setuptools

from reframe import VERSION

with open('README.md') as read_me:
    long_description = ''.join(read_me.readlines()[15:])

setuptools.setup(
    name='ReFrame-HPC',
    version=VERSION,
    author='CSCS Swiss National Supercomputing Center',
    description='ReFrame is a framework for writing regression tests '
                'for HPC systems',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/eth-cscs/reframe',
    license='BSD 3-Clause',
    packages=setuptools.find_namespace_packages(
        include=['reframe', 'reframe.*']
    ),
    package_data={'reframe': ['schemas/*']},
    include_package_data=True,
    install_requires=['jsonschema'],
    python_requires='>=3.6',
    scripts=['bin/reframe'],
    classifiers=(
        'Development Status :: 5 - Production/Stable',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'License :: OSI Approved :: BSD License',
        'Operating System :: MacOS',
        'Operating System :: POSIX :: Linux',
        'Environment :: Console'
    ),
)
