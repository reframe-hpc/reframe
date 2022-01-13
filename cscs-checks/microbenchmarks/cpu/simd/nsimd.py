# Copyright 2016-2021 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe as rfm
import reframe.utility.sanity as sn


@rfm.simple_test
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
    bench_name = parameter(['mul.avx2.f32.cpp', 'mul.avx2.f64.cpp'])

    valid_systems = ['dom:mc', 'dom:gpu', 'eiger:mc']
    valid_prog_environs = ['PrgEnv-gnu']
    build_system = 'SingleSource'

    # c++ test code generated with:
    #   python3 egg/hatch.py --benches --simd avx2 -m mul
    # and benches.hpp copied from:
    #   https://github.com/agenium-scale/nsimd/blob/master/benches/
    sourcesdir = 'benches'
    modules = ['nsimd', 'googlebenchmark', 'sleef', 'MIPP']
    maintainers = ['JG']
    exclusive = True
    num_tasks = 1
    num_tasks_per_node = 1
    num_cpus_per_task = 1
    num_tasks_per_core = 1
    use_multithreading = False

    allrefs = {'haswell': {'mul.avx2.f32.cpp': 7.8, 'mul.avx2.f64.cpp': 4.6},
               'broadwell': {'mul.avx2.f32.cpp': 7.4, 'mul.avx2.f64.cpp': 3.7},
               'zen2': {'mul.avx2.f32.cpp': 12., 'mul.avx2.f64.cpp': 7.6}}

    @run_after('init')
    def set_descr(self):
        self.descr = f'testing {self.bench_name}'

    @run_before('compile')
    def setup_build(self):
        self.sourcepath = self.bench_name
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

    @run_before('run')
    def prepare_run(self):
        self.skip_if_no_procinfo()
        self.executable = f'NsimdTest_{self.bench_name}'.replace('.', '_')
        self.variables = {
            'CRAYPE_LINK_TYPE': 'dynamic',
            'OMP_NUM_THREADS': str(self.num_cpus_per_task),
        }

        # Setup the reference
        proc = self.current_partition.processor
        if proc.info:
            perf_var = self.allrefs[proc.arch][self.bench_name]
            self.reference['*'] = (perf_var, -0.2, None)

    @sanity_function
    def validate_benchmark(self):
        return sn.assert_found(f'^Running {self.executable}', self.stderr)

    @performance_function('n/a')
    def speedup(self):
        regex = r'^\S+(f32|f64)\s+(\S+) ns\s+'
        slowest = sn.max(sn.extractall(regex, self.stdout, 2, float))
        fastest = sn.min(sn.extractall(regex, self.stdout, 2, float))
        return sn.round(slowest / fastest, 3)
