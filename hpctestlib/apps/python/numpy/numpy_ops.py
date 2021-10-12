# Copyright 2016-2021 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe as rfm
import reframe.utility.sanity as sn


class numpy_ops_check(rfm.RunOnlyRegressionTest, pin_prefix=True):
    '''NumPy basic operations test.

    `NumPy <https://numpy.org/>`__ is the fundamental package for scientific
    computing in Python.
    It provides a multidimensional array object, various derived objects
    (such as masked arrays and matrices), and an assortment of routines
    for fast operations on arrays, including mathematical, logical, shape
    manipulation, sorting, selecting, I/O, discrete Fourier transforms,
    basic linear algebra, basic statistical operations, random simulation
    and much more.

    This test test performs some fundamental NumPy linear algebra operations
    (matrix product, SVD, Cholesky decomposition, eigendecomposition, and
    inverse matrix calculation) and users the execution time as a performance
    metric. The default assumption is that NumPy is already installed on the
    currest system.
    '''

    #: See :attr:`~reframe.core.pipeline.RegressionTest.executable`.
    #:
    #: :required: No
    executable = 'python'

    #: See :attr:`~reframe.core.pipeline.RegressionTest.executable_opts`.
    #:
    #: :required: No
    executable_opts = ['np_ops.py']

    #: See :attr:`~reframe.core.pipeline.RegressionTest.descr`.
    #:
    #: :required: No
    descr = 'Test NumPy operations: dot, svd, cholesky, eigen and inv'

    @performance_function('seconds', perf_key='dot')
    def time_dot(self):
        return sn.extractsingle(
            r'^Dotted two 4096x4096 matrices in\s+(?P<dot>\S+)\s+s',
            self.stdout, 'dot', float)

    @performance_function('seconds', perf_key='svd')
    def time_svd(self):
        return sn.extractsingle(
            r'^SVD of a 2048x1024 matrix in\s+(?P<svd>\S+)\s+s',
            self.stdout, 'svd', float)

    @performance_function('seconds', perf_key='cholesky')
    def time_cholesky(self):
        return sn.extractsingle(
            r'^Cholesky decomposition of a 2048x2048 matrix in'
            r'\s+(?P<cholesky>\S+)\s+s',
            self.stdout, 'cholesky', float)

    @performance_function('seconds', perf_key='eigendec')
    def time_eigendec(self):
        return sn.extractsingle(
            r'^Eigendecomposition of a 2048x2048 matrix in'
            r'\s+(?P<eigendec>\S+)\s+s',
            self.stdout, 'eigendec', float)

    @performance_function('seconds', perf_key='inv')
    def time_inv(self):
        return sn.extractsingle(
            r'^Inversion of a 2048x2048 matrix in\s+(?P<inv>\S+)\s+s',
            self.stdout, 'inv', float)

    @sanity_function
    def assert_numpy_version(self):
        return sn.assert_found(r'Numpy version:\s+\S+', self.stdout)
