# Copyright 2016-2021 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe as rfm
import reframe.utility.sanity as sn


class Numpy_BaseCheck(rfm.RunOnlyRegressionTest, pin_prefix=True):
    '''Base class for the NumPy Test.
    NumPy is the fundamental package for scientific computing in Python.
    It is a Python library that provides a multidimensional array object,
    various derived objects (such as masked arrays and matrices), and an
    assortment of routines for fast operations on arrays, including
    mathematical, logical, shape manipulation, sorting, selecting, I/O,
    discrete Fourier transforms, basic linear algebra, basic statistical
    operations, random simulation and much more (see numpy.org).

    The presented abstract run-only class checks the numpy perfomance.
    This test checks whether some basic operations (such as matrix product,
    SVD decomposition, Cholesky decomposition, eigendecomposition, and
    inverse matrix calculation) are performed, and also checks the
    execution time of these operations. The default assumption is that
    NumPy is already installed on the device under test.
    '''

    num_tasks_per_node = required

    executable = 'python'
    executable_opts = ['np_ops.py']

    @run_after('init')
    def set_description(self):
        self.mydescr = 'Test a few typical numpy operations'

    @performance_function('seconds', perf_key='dot')
    def set_perf_dot(self):
        return sn.extractsingle(
            r'^Dotted two 4096x4096 matrices in\s+(?P<dot>\S+)\s+s',
            self.stdout, 'dot', float)

    @performance_function('seconds', perf_key='svd')
    def set_perf_svd(self):
        return sn.extractsingle(
            r'^SVD of a 2048x1024 matrix in\s+(?P<svd>\S+)\s+s',
            self.stdout, 'svd', float)

    @performance_function('seconds', perf_key='cholesky')
    def set_perf_cholesky(self):
        return sn.extractsingle(
            r'^Cholesky decomposition of a 2048x2048 matrix in'
            r'\s+(?P<cholesky>\S+)\s+s',
            self.stdout, 'cholesky', float)

    @performance_function('seconds', perf_key='eigendec')
    def set_perf_eigendec(self):
        return sn.extractsingle(
            r'^Eigendecomposition of a 2048x2048 matrix in'
            r'\s+(?P<eigendec>\S+)\s+s',
            self.stdout, 'eigendec', float)

    @performance_function('seconds', perf_key='inv')
    def set_perf_inv(self):
        return sn.extractsingle(
            r'^Inversion of a 2048x2048 matrix in\s+(?P<inv>\S+)\s+s',
            self.stdout, 'inv', float)

    @sanity_function
    def assert_energy_readout(self):
        return sn.assert_found(r'Numpy version:\s+\S+', self.stdout)
