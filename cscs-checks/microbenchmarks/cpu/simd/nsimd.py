# Copyright 2016-2020 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe as rfm
import reframe.utility.sanity as sn


@rfm.parameterized_test(['mul.avx2.f32.cpp'], ['mul.avx2.f64.cpp'])
class NsimdTest(rfm.RegressionTest):
    '''
    Testing https://github.com/agenium-scale/nsimd.git

    > cat rfm_NsimdTest_mul_avx2_f32_cpp_job.out
        ---------------------------------------------------------------------
        Benchmark                           Time             CPU   Iterations
        ---------------------------------------------------------------------
        nsimd_cpu_mul/f32                94.4 ns         94.4 ns      7416152
        nsimd_avx2_mul/f32               83.5 ns         83.5 ns      8387640
        nsimd_avx2_mul_unroll2/f32       56.8 ns         56.8 ns     12329852
        nsimd_avx2_mul_unroll3/f32       56.0 ns         56.0 ns     12491950
        nsimd_avx2_mul_unroll4/f32       39.9 ns         39.9 ns     17561039
        std_mul/f32                       491 ns          491 ns      1425407
        MIPP_mul/f32                     83.5 ns         83.5 ns      8386285

    Example PERFORMANCE REPORT
    --------------------------
    NsimdTest_mul_avx2_f32_cpp
    - eiger:mc
       - builtin
          * num_tasks: 1
          * speedup: 12.306 x (ns)
    '''
    def __init__(self, testname):
        self.valid_systems = ['dom:mc', 'dom:gpu', 'eiger:mc']
        self.valid_prog_environs = ['builtin']
        self.descr = f'testing {testname}'
        self.build_system = 'SingleSource'
        self.testname = testname
        # c++ test code generated with:
        #   python3 egg/hatch.py --benches --simd avx2 -m mul
        self.sourcesdir = 'benches'
        self.sourcepath = self.testname
        self.executable = f'{testname}.exe'
        self.modules = ['nsimd', 'googlebenchmark', 'sleef', 'MIPP']
        self.build_system.cxxflags = [
            '-std=c++14', '-O3', '-DNDEBUG', '-dynamic',
            '-mavx2', '-DAVX2', '-mfma', '-DFMA',
            '-I$EBROOTNSIMD/include',
            '-I$EBROOTGOOGLEBENCHMARK/include',
        ]
        self.build_system.ldflags = [
            '-L$EBROOTNSIMD/lib', '-L$EBROOTGOOGLEBENCHMARK/lib64',
            '-L$EBROOTSLEEF/lib', '-lnsimd_avx2', '-lbenchmark', '-lpthread',
            '-lsleef'
        ]
        self.maintainers = ['JG']
        self.exclusive = True
        self.num_tasks = 1
        self.num_tasks_per_node = 1
        self.num_cpus_per_task = 1
        self.num_tasks_per_core = 1
        self.use_multithreading = False
        self.variables = {
            'CRAYPE_LINK_TYPE': 'dynamic',
            'OMP_NUM_THREADS': str(self.num_cpus_per_task),
        }
        self.sanity_patterns = sn.assert_found('^Running %s' % self.executable,
                                               self.stderr)
        self.perf_patterns = {
            'speedup': self.speedup,
        }
        reference = {'hwl': {'mul.avx2.f32.cpp': 7.8, 'mul.avx2.f64.cpp': 4.6},
                     'bwl': {'mul.avx2.f32.cpp': 7.4, 'mul.avx2.f64.cpp': 3.7},
                     'eig': {'mul.avx2.f32.cpp': 12., 'mul.avx2.f64.cpp': 7.6}}
        self.reference = {
            'dom:gpu': {
                'speedup': (reference['hwl'][testname], -0.2, None, 'x (ns)')
            },
            'dom:mc': {
                'speedup': (reference['bwl'][testname], -0.2, None, 'x (ns)')
            },
            'eiger:mc': {
                'speedup': (reference['eig'][testname], -0.2, None, 'x (ns)')
            },
            '*': {
                'speedup': (1.0, None, None, 'x (ns)')
            }
        }

    @property
    @sn.sanity_function
    def speedup(self):
        regex = r'^\S+(f32|f64)\s+(\S+) ns\s+'
        slowest = sn.max(sn.extractall(regex, self.stdout, 2, float))
        fastest = sn.min(sn.extractall(regex, self.stdout, 2, float))
        return sn.round(slowest / fastest, 3)
