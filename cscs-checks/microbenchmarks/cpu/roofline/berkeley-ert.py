# Copyright 2016-2021 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import os
import reframe as rfm
import reframe.utility.sanity as sn
import reframe.utility.udeps as udeps

# {{{ readme:
# This check runs and plots the ERT for different cpus. For example:
# - for Intel Haswell (HWL):
# ('HWL_Ert_Check(Ert_Base_RunCheck)')
#   -> ('HWL_Ert_BaseCheck_{mpi}_{flops}(Ert_BaseCheck)') * {repeat} times.
# - for Intel Broadwell (BWL):
# ('BWL_Ert_Check(Ert_Base_RunCheck)')
#   -> ('BWL_Ert_BaseCheck_{mpi}_{flops}(Ert_BaseCheck)') * {repeat} times.
# For a complete simulation, uncomment the following lines:
# flops = [2**x for x in range(11)]  # maxrange=11 / 1:1024
# mpi_tasks_d = {
#     'broadwell': [1, 2, 3, 4, 6, 9, 12, 18, 36],
#     # mpi*omp -> [1 36| 2 18| 3 12| 4 9| 6 6| 9 4| 12 3| 18 2| 36 1]
#     # lscpu: L1d cache: 32K L1i cache: 32K L2 cache: 256K L3 cache: 46080K
#     # ---------------------------------------------------
#     'haswell': [1, 2, 3, 4, 6, 12]
#     # mpi*omp -> [1 12| 2 6| 3 4| 4 3| 6 2| 12 1]
#     # lscpu: L1d cache: 32K L1i cache: 32K L2 cache: 256K L3 cache: 30720K
#     # ---------------------------------------------------
# }
# The kernels are set in:
# > grep 'define KERNEL' Empirical_Roofline_Tool-1.1.0/Kernels/kernel1.h
# #define KERNEL1(a, b, c) ((a) = (b) + (c))
# #define KERNEL2(a, b, c) ((a) = (a) * (b) + (c))
# The flops are set in:
# > egrep "add .* flop" Kernels/kernel1.h
# #if (ERT_FLOP & 1) == 1 /* add 1 flop */
# #if (ERT_FLOP & 2) == 2 /* add 2 flops */
# #if (ERT_FLOP & 4) == 4 /* add 4 flops */
# #if (ERT_FLOP & 8) == 8 /* add 8 flops */
# #if (ERT_FLOP & 16) == 16 /* add 16 flops */
# #if (ERT_FLOP & 32) == 32 /* add 32 flops */
# #if (ERT_FLOP & 64) == 64 /* add 64 flops */
# #if (ERT_FLOP & 128) == 128 /* add 128 flops */
# #if (ERT_FLOP & 256) == 256 /* add 256 flops */
# #if (ERT_FLOP & 512) == 512 /* add 512 flops */
# #if (ERT_FLOP & 1024) == 1024 /* add 1024 flops */
# A typical output is:
# working_set_size*bytes_per_elem
#                        t
#                                  seconds  total_bytes  total_flops
#        1024            1         731.238         2048       131072
#        1024            2           7.705         4096       262144
#        1024            4          11.562         8192       524288
#        1024            8          22.242        16384      1048576
#        etc...
# }}}
repeat = 2
flops = [4]
mpi_tasks_d = {'broadwell': [18, 36], 'haswell': [6, 12]}


# {{{ class Ert_BaseCheck
class Ert_BaseCheck(rfm.RegressionTest):
    def __init__(self, mpi_task, flop):
        self.descr = f'Empirical Roofline Toolkit (Base for running)'
        # pe step
        # build step
        # run step
        # postprocess step
        # sanity step

    # {{{ hooks
    @rfm.run_before('compile')
    def set_compiler_flags_and_variables(self):
        self.sourcesdir = os.path.join(
            self.current_system.resourcesdir,
            "roofline",
            "cs-roofline-toolkit.git",
            "Empirical_Roofline_Tool-1.1.0",
        )
        self.sourcepath = 'Kernels/kernel1.cxx Drivers/driver1.cxx'
        self.build_system = 'SingleSource'
        flop = sn.getattr(self, 'flop')
        self.build_system.cppflags = [
            '-I./Kernels',
            f'-DERT_FLOP={flop}',
            '-DERT_ALIGN=32',
            '-DERT_MEMORY_MAX=1073741824',
            '-DERT_MPI=True',
            '-DERT_OPENMP=True',
            '-DERT_TRIALS_MIN=1',
            '-DERT_WORKING_SET_MIN=1',
            '-DERT_WSS_MULT=1.1',
            '-DERT_FP64',
            # '-DERT_INTEL',
        ]
        self.prgenv_flags = {
            'PrgEnv-gnu': ['-fopenmp', '-O3'],
        }
        self.build_system.cxxflags = \
            self.prgenv_flags[self.current_environ.name]
        self.variables = {
            # 'CRAYPE_LINK_TYPE': 'dynamic',
            'OMP_NUM_THREADS': str(self.num_cpus_per_task),
            'OMP_PROC_BIND': 'close',
            'OMP_PLACES': 'cores',
        }

    @rfm.run_before('run')
    def set_run_cmds(self):
        self.prerun_cmds += [f'for ii in `seq {repeat}`;do']
        self.executable_opts = ['&> try.00$ii']
        self.job.launcher.options = ['--cpu-bind=cores']  # verbose
        self.postrun_cmds += [
            'done',
            'cat try.00* | ./Scripts/preprocess.py > pre',
            './Scripts/maximum.py < pre > max',
            './Scripts/summary.py < max > sum',
        ]

    @rfm.run_before('sanity')
    def set_sanity(self):
        self.sanity_patterns = sn.all(
            [
                sn.assert_found(r'^fp64', 'try.001'),
                sn.assert_found(r'^fp64', 'max'),
                sn.assert_found(r'GFLOPs|DRAM', 'sum'),
                sn.assert_found('META_DATA', 'sum'),
            ]
        )
    # }}}
# }}}


# {{{ class Ert_Base_RunCheck
class Ert_Base_RunCheck(rfm.RunOnlyRegressionTest):
    def __init__(self):
        self.descr = f'Empirical Roofline Toolkit (Base for plotting)'
        self.roofline_script1_fname = 'roofline.py'
        self.roofline_script1 = f'./Scripts/{self.roofline_script1_fname}'
        self.roofline_script2 = './ert_cscs.py'
        self.roofline_out_script1 = 'o.roofline'
        self.roofline_summary = 'sum'
        self.executable = f'cat'
        self.executable_opts = [
            r'*.sum', r'|', 'python3', self.roofline_script1_fname,
            '&> ', self.roofline_out_script1
        ]
        self.postrun_cmds = [
            self.roofline_script2,
            'gnuplot roofline.gnuplot',  # name hardcoded in the script
            'file roofline.ps',  # formats originally avail: gnuplot, json, tex
        ]
        # {{{ sanity_patterns
        self.sanity_patterns = sn.all(
            [
                sn.assert_found(r'GFLOPs EMP', self.roofline_out_script1),
                sn.assert_found(r'DRAM EMP', self.roofline_out_script1),
                sn.assert_found('Empirical roofline graph:', self.stdout),
                sn.assert_found(r'.ps: PostScript', self.stdout),
            ]
        )
        # }}}

        # {{{ performance
        # {{{
        # Typical performance report looks like:
        # --------------------------------------
        # > cat o.roofline
        # 84.87 FP64 GFLOPs EMP <----
        #  META_DATA
        #    FLOPS          8
        #    MPI_PROCS      2
        #    OPENMP_THREADS 64
        #
        # 392.47 L1 EMP         <---- + L2, L3, L4
        #  137.94 DRAM EMP      <----
        #  META_DATA
        #    FLOPS          1
        #    MPI_PROCS      2
        #    OPENMP_THREADS 64
        # }}}
        regex_gflops = r'(\S+)\sFP64 GFLOPs EMP'
        regex_L1bw = r'(\S+)\sL1 EMP'
        regex_DRAMbw = r'(\S+)\sDRAM EMP'
        gflops = sn.extractsingle(regex_gflops, self.roofline_out_script1, 1,
                                  float)
        DRAMbw = sn.extractsingle(regex_DRAMbw, self.roofline_out_script1, 1,
                                  float)
        self.perf_patterns = {
            'gflops': gflops,
            'DRAMbw': DRAMbw,
        }
        # }}}
# }}}


# {{{ Intel Haswell
# {{{ class HWL_Ert_BaseCheck
@rfm.parameterized_test(*[[mpi_task, flop]
                          for mpi_task in mpi_tasks_d['haswell']
                          for flop in flops
                          ])
class HWL_Ert_BaseCheck(Ert_BaseCheck):
    def __init__(self, mpi_task, flop):
        # {{{ pe
        self.descr = 'Empirical Roofline Toolkit (Base for Haswell)'
        self.valid_systems = ['dom:gpu']
        self.valid_prog_environs = ['PrgEnv-gnu']
        # }}}

        # {{{ build
        self.flop = flop
        # }}}

        # {{{ run
        self.num_tasks = mpi_task
        self.num_tasks_per_node = mpi_task
        self.num_cpus_per_task = 12 // mpi_task
        self.num_tasks_per_core = 1
        self.use_multithreading = False
        self.exclusive = True
        # Assuming repeat=2, time can be adjusted as:
        if flop >= 512:
            self.time_limit = '20m'
        else:
            self.time_limit = '10m'
        # The other steps are in the base class
        # }}}
# }}}


# {{{ HWL_Ert_Check
@rfm.simple_test
# class HWL_Ert_Check(rfm.RunOnlyRegressionTest):
class HWL_Ert_Check(Ert_Base_RunCheck):
    """
    The Empirical Roofline Tool, ERT, automatically generates roofline data.
    https://bitbucket.org/berkeleylab/cs-roofline-toolkit/
    This class depends on the HWL_Ert_BaseCheck class.
    """

    def __init__(self):
        super().__init__()
        self.descr = 'Final step of the Roofline test (Intel Haswell cpu)'
        self.valid_systems = ['dom:login']
        self.valid_prog_environs = ['PrgEnv-gnu']
        self.maintainers = ['JG']
        self.tags = {'cpu'}
        self.sourcesdir = None
        # self.modules = ['gnuplot']
        for ii in mpi_tasks_d['haswell']:
            for jj in flops:
                self.depends_on(f'HWL_Ert_BaseCheck_{ii}_{jj}', udeps.by_env)

        # {{{ performance
        # Reference roofline boundaries for Intel Haswell cpu:
        ref_GFLOPs = 330.0
        ref_DRAMbw = 53.0
        self.reference = {
            '*': {
                'gflops': (ref_GFLOPs, None, None, 'GF/s'),
                'DRAMbw': (ref_DRAMbw, None, None, 'GB/s'),
            }
        }
        # }}}

    # {{{ hooks
    @rfm.require_deps
    def prepare_logs(self, HWL_Ert_BaseCheck):
        """
        get all the summary files from the compute jobs for postprocessing
        """
        job_out = 'sum'
        for ii in mpi_tasks_d['haswell']:
            for jj in flops:
                dir_fullpath = self.getdep(
                    f'HWL_Ert_BaseCheck_{ii}_{jj}', part='gpu'
                ).stagedir
                dir_basename = dir_fullpath.split("/")[-1]
                self.prerun_cmds.append(
                    f'ln -s {dir_fullpath}/{job_out} {dir_basename}.{job_out}'
                )

        self.prerun_cmds.append(
            f'ln -s {dir_fullpath}/{self.roofline_script1}'
        )
        self.prerun_cmds.append(
            f'ln -s {dir_fullpath}/{self.roofline_script2}'
        )
        self.prerun_cmds.append(f'ln -s {dir_fullpath}/Plot/')
    # }}}
# }}}
# }}}


# {{{ Intel Broadwell
# {{{ class BWL_Ert_BaseCheck
@rfm.parameterized_test(*[[mpi_task, flop]
                          for mpi_task in mpi_tasks_d['broadwell']
                          for flop in flops
                          ])
class BWL_Ert_BaseCheck(Ert_BaseCheck):
    def __init__(self, mpi_task, flop):
        # {{{ pe
        self.descr = 'Empirical Roofline Toolkit (Base for Broadwell)'
        self.valid_systems = ['dom:mc']
        self.valid_prog_environs = ['PrgEnv-gnu']
        # }}}

        # {{{ build
        self.flop = flop
        # }}}

        # {{{ run
        self.num_tasks = mpi_task
        self.num_tasks_per_node = mpi_task
        self.num_cpus_per_task = 36 // mpi_task
        self.num_tasks_per_core = 1
        self.use_multithreading = False
        self.exclusive = True
        # Assuming repeat=2, time can be adjusted as:
        if flop >= 64 or mpi_task >= 12:
            self.time_limit = '20m'
        else:
            self.time_limit = '10m'
        # The other steps are in the base class
        # }}}
# }}}


# {{{ BWL_Ert_Check
@rfm.simple_test
class BWL_Ert_Check(Ert_Base_RunCheck):
    """
    The Empirical Roofline Tool, ERT, automatically generates roofline data.
    https://bitbucket.org/berkeleylab/cs-roofline-toolkit/
    This class depends on the BWL_Ert_BaseCheck class.
    """

    def __init__(self):
        super().__init__()
        self.descr = 'Final step of the Roofline test (Intel Broadwell cpu)'
        self.valid_systems = ['dom:login']
        self.valid_prog_environs = ['PrgEnv-gnu']
        self.maintainers = ['JG']
        self.tags = {'cpu'}
        self.sourcesdir = None
        # self.modules = ['gnuplot']
        for ii in mpi_tasks_d['broadwell']:
            for jj in flops:
                self.depends_on(f'BWL_Ert_BaseCheck_{ii}_{jj}', udeps.by_env)

        # {{{ performance
        # Reference roofline boundaries for Intel Broadwell cpu:
        ref_GFLOPs = 817.0
        ref_DRAMbw = 106.0
        self.reference = {
            '*': {
                'gflops': (ref_GFLOPs, None, None, 'GF/s'),
                'DRAMbw': (ref_DRAMbw, None, None, 'GB/s'),
            }
        }
        # }}}

    # {{{ hooks
    @rfm.require_deps
    def prepare_logs(self, HWL_Ert_BaseCheck):
        """
        get all the summary files from the compute jobs for postprocessing
        """
        job_out = 'sum'
        for ii in mpi_tasks_d['broadwell']:
            for jj in flops:
                dir_fullpath = self.getdep(
                    f'BWL_Ert_BaseCheck_{ii}_{jj}', part='mc'
                ).stagedir
                dir_basename = dir_fullpath.split("/")[-1]
                self.prerun_cmds.append(
                    f'ln -s {dir_fullpath}/{job_out} {dir_basename}.{job_out}'
                )

        self.prerun_cmds.append(
            f'ln -s {dir_fullpath}/{self.roofline_script1}'
        )
        self.prerun_cmds.append(
            f'ln -s {dir_fullpath}/{self.roofline_script2}'
        )
        self.prerun_cmds.append(f'ln -s {dir_fullpath}/Plot/')
    # }}}
# }}}
# }}}
