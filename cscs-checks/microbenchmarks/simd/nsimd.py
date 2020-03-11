# Copyright 2016-2020 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe as rfm
import reframe.utility.sanity as sn


@rfm.simple_test
class NsimdTest(rfm.RegressionTest):
    '''
    Testing https://github.com/agenium-scale/nsimd
    Example job (with --performance-report) will look like:
    ----------------------------------------------------------------------
    Benchmark                            Time             CPU   Iterations
    ----------------------------------------------------------------------
    nsimd_cpu_sqrt/f64                 747 ns          747 ns       935399
    nsimd_avx2_sqrt/f64               1191 ns         1191 ns       589402
    nsimd_avx2_sqrt_unroll2/f64       1222 ns         1222 ns       568000
    nsimd_avx2_sqrt_unroll3/f64       1187 ns         1187 ns       584756
    nsimd_avx2_sqrt_unroll4/f64       1187 ns         1187 ns       590104
    Sleef_Sleef_sqrtd4_avx2/f64       1174 ns         1174 ns       596736
    std_sqrt/f64                      3149 ns         3149 ns       222585
    MIPP_sqrt/f64                     1188 ns         1188 ns       589450

    > reframe --system dom:mc -p PrgEnv-gnu -r -c nsimd.py
        PERFORMANCE REPORT
        -----------------------------------------------------------------------
        NsimdTestt
        - dom:mc
           - PrgEnv-gnu
            * speedup: 3.827 x (ns)
    > reframe --system dom:gpu -p PrgEnv-gnu -r -c nsimd.py
        PERFORMANCE REPORT
        -----------------------------------------------------------------------
        NsimdTestt
        - dom:gpu
           - PrgEnv-gnu
            * speedup: 4.254 x (ns)

    > Test code is:
        # std_sqrt (sqrt.avx2.f64.cpp:308)
        *(_r + i) = std::sqrt(*(_0 + i));

        # nsimd_cpu_sqrt (sqrt.avx2.f64.cpp:104)
        return _mm256_sqrt_pd(a0);
        nsimd::storea(_r + i,
                      nsimd::sqrt(nsimd::loada(_0 + i, f64(), nsimd::cpu()),
                                  f64(),
                                  nsimd::cpu()),
                      f64(),
                      nsimd::cpu()  );

        # nsimd_avx2_sqrt (sqrt.avx2.f64.cpp:138)
        _mm256_sqrt_pd (avxintrin.h:1039)

        # nsimd_avx2_sqrt_unroll2
        nsimd::storea(_r + i, nsimd::sqrt(nsimd::loada<nsimd::pack<f64, 2>
                      >(_0 + i)));

        # nsimd_avx2_sqrt_unroll3
        nsimd::storea(_r + i, nsimd::sqrt(nsimd::loada<nsimd::pack<f64, 3>
                      >(_0 + i)));

        # nsimd_avx2_sqrt_unroll4
        nsimd::storea(_r + i, nsimd::sqrt(nsimd::loada<nsimd::pack<f64, 4>
                      >(_0 + i)));

        # Sleef_Sleef_sqrtd4_avx2
        nsimd::storea(_r + i, Sleef_sqrtd4_avx2(nsimd::loada(_0 + i, f64())),
                      f64());

        # MIPP_sqrt
        mipp::store(_r + i, mipp::sqrt<f64>(mipp::load<f64>(_0 + i)));
    '''
    def __init__(self):
        self.valid_systems = ['dom:mc', 'dom:gpu']
        self.valid_prog_environs = ['builtin']
        self.descr = 'sqrt.avx2.f64 example'
        self.build_system = 'SingleSource'
        self.testname = 'sqrt.avx2.f64'
        self.sourcesdir = None
        self.srcdir = 'benches/cxx_adv'
        self.sourcepath = '%s/%s.cpp' % (self.srcdir, self.testname)
        self.prebuild_cmd = [
            'tar xf $EBROOTNSIMD/benches.tar benches/benches.hpp',
            'tar xf $EBROOTNSIMD/benches.tar %s' % self.sourcepath,
        ]
        self.executable = '%s.exe' % self.testname
        self.modules = ['nsimd/579084-CrayGNU-19.06']
        self.build_system.cxxflags = [
            '-std=c++14', '-O3', '-DNDEBUG', '-dynamic',
            '-mavx2', '-DAVX2', '-mfma', '-DFMA',
            '-I$EBROOTNSIMD/include/include',
            '-I$EBROOTGOOGLEBENCHMARK/include',
        ]
        self.build_system.ldflags = [
            '-L$EBROOTNSIMD/lib', '-L$EBROOTGOOGLEBENCHMARK/lib64',
            '-L$EBROOTSLEEF/lib', '-lnsimd_x86_64', '-lbenchmark', '-lpthread',
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
        self.reference = {
            'dom:gpu': {
                'speedup': (4.2, -0.2, 0.2, 'x (ns)')
            },
            'dom:mc': {
                'speedup': (3.8, -0.2, 0.2, 'x (ns)')
            },
            '*': {
                'speedup': (1.0, None, None, 'x (ns)')
            }
        }

    @property
    @sn.sanity_function
    def speedup(self):
        # comparing fast and slow timings:
        slow = sn.extractsingle(r'^std_sqrt/f64\s+(?P<slow>\d+) ns',
                                self.stdout, 'slow', int)
        fast = sn.extractsingle(r'^nsimd_cpu_sqrt/f64\s+(?P<fast>\d+) ns',
                                self.stdout, 'fast', int)
        sp = sn.round(slow / fast, 3)
        return sp
