# Copyright 2016-2021 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import os
from shutil import which
import reframe as rfm
import reframe.utility.sanity as sn
import reframe.utility.udeps as udeps

# NOTE: do not run this check with --system (because of deps)
ert_precisions = ['ERT_FP64']
ert_repeat = 1
ert_flops = [4]
# NOTE: uncomment for full search:
# ert_flops = [2**x for x in range(11)]  # maxrange=11 / 1:1024
cpu_specs = {
    'HWL': {
        'ref_GFLOPs': 330.0,
        'ref_DRAMbw': 53.0,
        'mpi_tasks': [6, 12],
        # NOTE: uncomment for full search:
        # 'mpi_tasks': [1, 2, 3, 4, 6, 8, 12],
        # NOTE: lscpu: L1d cache: 32K L1i cache: 32K
        #               L2 cache: 256K L3 cache: 30720K
    },
    'BWL': {
        'ref_GFLOPs': 330.0,
        'ref_DRAMbw': 53.0,
        'mpi_tasks': [18, 36],
        # NOTE: uncomment for full search:
        # 'mpi_tasks': [1, 2, 3, 4, 6, 9, 12, 18, 36],
        # NOTE: lscpu: L1d cache: 32K L1i cache: 32K
        #               L2 cache: 256K L3 cache: 46080K
    },
}


# {{{ class RunErt_Base
class RunErt_Base(rfm.RegressionTest):
    # {{{
    """
    This check runs and plots the ERT for different cpus.
    The kernels are set as:
    > grep 'define KERNEL' Empirical_Roofline_Tool-1.1.0/Kernels/kernel1.h
    #define KERNEL1(a, b, c) ((a) = (b) + (c))
    #define KERNEL2(a, b, c) ((a) = (a) * (b) + (c))

    The flops are set in:
    > egrep "add .* flop" Kernels/kernel1.h
    #if (ERT_FLOP & 1) == 1 /* add 1 flop */
    #if (ERT_FLOP & 2) == 2 /* add 2 flops */
    #if (ERT_FLOP & 4) == 4 /* add 4 flops */
    #if (ERT_FLOP & 8) == 8 /* add 8 flops */
    #if (ERT_FLOP & 16) == 16 /* add 16 flops */
    #if (ERT_FLOP & 32) == 32 /* add 32 flops */
    #if (ERT_FLOP & 64) == 64 /* add 64 flops */
    #if (ERT_FLOP & 128) == 128 /* add 128 flops */
    #if (ERT_FLOP & 256) == 256 /* add 256 flops */
    #if (ERT_FLOP & 512) == 512 /* add 512 flops */
    #if (ERT_FLOP & 1024) == 1024 /* add 1024 flops */

    A typical output is:
    working_set_size*bytes_per_elem
                           t
                                     seconds  total_bytes  total_flops
           1024            1         731.238         2048       131072
           1024            2           7.705         4096       262144
           1024            4          11.562         8192       524288
           1024            8          22.242        16384      1048576
           etc...
    """
    # }}}
    def __init__(self):
        self.descr = f'Empirical Roofline Toolkit (Base for building/running)'
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
            'roofline',
            'cs-roofline-toolkit.git',
            'Empirical_Roofline_Tool-1.1.0',
        )
        self.readonly_files = [
            'Batch',
            'Config',
            'Drivers',
            'ert',
            'ert_cscs.py',
            'ERT_Users_Manual.pdf',
            'Kernels',
            'Plot',
            'Python',
            'README.md',
            'Results',
            'Scripts',
        ]
        # Using a sourcepath trick here to remain close to the way building is
        # executed in the official repo script (no makefile provided):
        # https://bitbucket.org/berkeleylab/cs-roofline-toolkit/src/master/
        # Empirical_Roofline_Tool-1.1.0/Python/ert_core.py#lines-279
        self.sourcepath = 'Kernels/kernel1.cxx Drivers/driver1.cxx'
        self.build_system = 'SingleSource'
        # get all parameters:
        ert_trials_min = sn.getattr(self, 'ert_trials_min')
        ert_precision = sn.getattr(self, 'ert_precision')
        ert_flop = sn.getattr(self, 'ert_flop')
        self.build_system.cppflags = [
            '-I./Kernels',
            f'-DERT_FLOP={ert_flop}',
            '-DERT_ALIGN=32',
            '-DERT_MEMORY_MAX=1073741824',
            '-DERT_MPI=True',
            '-DERT_OPENMP=True',
            '-DERT_WORKING_SET_MIN=1',
            '-DERT_WSS_MULT=1.1',
            f'-D{ert_precision}',
            f'-DERT_TRIALS_MIN={ert_trials_min}',
            # keeping as reminder
            # '-DERT_INTEL',
        ]
        self.prgenv_flags = {
            'PrgEnv-gnu': ['-fopenmp', '-O3'],
        }
        envname = self.current_environ.name
        self.build_system.cxxflags = self.prgenv_flags[envname]
        self.prebuild_cmds = ['module list', 'which gcc']

    @rfm.run_before('run')
    def set_run_cmds(self):
        self.variables = {
            'OMP_NUM_THREADS': str(self.num_cpus_per_task),
            'OMP_PROC_BIND': 'close',
            'OMP_PLACES': 'cores',
        }
        cmd = self.job.launcher.run_command(self.job)
        self.executable_opts = ['&> try.00']
        self.job.launcher.options = ['--cpu-bind=cores']  # verbose
        self.postrun_cmds += [
            f'{cmd} {self.job.launcher.options[0]} {self.executable} '
            f'&> try.0{i}'
            for i in range(1, ert_repeat)
        ]

    @rfm.run_before('run')
    def set_postrun_cmds(self):
        self.postrun_cmds += [
            'cat try.* | ./Scripts/preprocess.py > pre',
            './Scripts/maximum.py < pre > max',
            './Scripts/summary.py < max > sum',
        ]

    @rfm.run_before('sanity')
    def set_sanity(self):
        self.sanity_patterns = sn.all(
            [
                sn.assert_found(r'^fp64', 'try.00'),
                sn.assert_found(r'^fp64', 'max'),
                sn.assert_found(r'GFLOPs|DRAM', 'sum'),
                sn.assert_found('META_DATA', 'sum'),
            ]
        )
    # }}}
# }}}


# {{{ class PlotErt_Base
class PlotErt_Base(rfm.RunOnlyRegressionTest):
    def __init__(self):
        self.descr = f'Empirical Roofline Toolkit (Base for plotting)'
        self.roofline_script1_fname = 'roofline.py'
        self.roofline_script1 = f'./Scripts/{self.roofline_script1_fname}'
        self.roofline_script2 = './ert_cscs.py'
        self.roofline_out_script1 = 'o.roofline'
        self.roofline_summary = 'sum'
        self.executable = f'cat'
        self.executable_opts = [
            r'*.sum',
            r'|',
            'python3',
            self.roofline_script1_fname,
            '&> ',
            self.roofline_out_script1,
        ]

        # {{{ sanity_patterns
        self.sanity_patterns = sn.all(
            [
                sn.assert_found(r'GFLOPs EMP', self.roofline_out_script1),
                sn.assert_found(r'DRAM EMP', self.roofline_out_script1),
                sn.assert_found('Empirical roofline graph:', self.stdout),
            ]
        )
        # }}}

        # {{{ performance
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

    # {{{ hooks
    @rfm.run_before('run')
    def check_gnuplot(self):
        gnuplot = which('gnuplot')
        if gnuplot is None:
            self.postrun_cmds = [
                self.roofline_script2,
                '# gnuplot roofline.gnuplot',  # name hardcoded in the script
                '# file roofline.ps',  # available formats: gnuplot, json, tex
            ]
        else:
            self.postrun_cmds = [
                self.roofline_script2,
                'gnuplot roofline.gnuplot',
                'file roofline.ps',
            ]
    # }}}
# }}}


# {{{ Intel Haswell
# {{{ HWL_RunErt
@rfm.parameterized_test(
    *[
        [ert_precision, ert_flop, ert_mpi_task]
        for ert_precision in ert_precisions
        for ert_flop in ert_flops
        for ert_mpi_task in cpu_specs['HWL']['mpi_tasks']
    ]
)
class HWL_RunErt(RunErt_Base):
    def __init__(self, ert_precision, ert_flop, ert_mpi_task):
        # {{{ pe
        cpu = 'HWL'
        self.descr = f'Collect ERT data from INTEL {cpu}'
        self.valid_systems = ['dom:gpu']
        self.valid_prog_environs = ['PrgEnv-gnu']
        # }}}

        # {{{ build
        self.ert_trials_min = 1
        self.ert_precision = ert_precision
        self.ert_flop = ert_flop
        # }}}

        # {{{ run
        self.num_tasks = ert_mpi_task
        self.num_tasks_per_node = ert_mpi_task
        self.num_cpus_per_task = 12 // ert_mpi_task
        # NOTE: mpi*openmp -> [1 12| 2 6| 3 4| 4 3| 6 2| 12 1]
        self.num_tasks_per_core = 1
        self.use_multithreading = False
        self.exclusive = True
        # Assuming ert_repeat=2, time can be adjusted as:
        if ert_flop >= 512:
            self.time_limit = '20m'
        else:
            self.time_limit = '10m'
        # The other steps are in the base class
        # }}}
# }}}


# {{{ HWL_PlotErt
@rfm.simple_test
class HWL_PlotErt(PlotErt_Base):
    """
    The Empirical Roofline Tool, ERT, automatically generates roofline data.
    https://bitbucket.org/berkeleylab/cs-roofline-toolkit/
    This class depends on the HWL_RunErt class.
    It can be run with: -n HWL_PlotErt -r
    """

    def __init__(self):
        super().__init__()
        cpu = 'HWL'
        self.descr = f'Plot ERT data on the Roofline chart (INTEL {cpu})'
        self.valid_systems = ['dom:login']
        self.valid_prog_environs = ['PrgEnv-gnu']
        self.maintainers = ['JG']
        self.tags = {'cpu'}
        self.sourcesdir = None
        # gnuplot already installed as rpm on dom but keeping as reminder
        # self.modules = ['gnuplot']
        self.dep_name = f'{cpu}_RunErt'
        for ii in ert_precisions:
            for jj in ert_flops:
                for kk in cpu_specs[cpu]['mpi_tasks']:
                    self.depends_on(f'{self.dep_name}_{ii}_{jj}_{kk}',
                                    udeps.by_env)

        # {{{ performance
        self.reference = {
            '*': {
                'gflops': (cpu_specs[cpu]['ref_GFLOPs'], None, None, 'GF/s'),
                'DRAMbw': (cpu_specs[cpu]['ref_DRAMbw'], None, None, 'GB/s'),
            }
        }
        # }}}

    # {{{ hooks
    @rfm.require_deps
    def prepare_logs(self, HWL_RunErt):
        """
        get all the summary files from the compute jobs for postprocessing
        """
        cpu = 'HWL'
        job_out = 'sum'
        for ii in ert_precisions:
            for jj in ert_flops:
                for kk in cpu_specs[cpu]['mpi_tasks']:
                    dir_fullpath = self.getdep(
                        f'{self.dep_name}_{ii}_{jj}_{kk}', part='gpu'
                    ).stagedir
                    dir_basename = dir_fullpath.split('/')[-1]
                    self.prerun_cmds.append(
                        f'ln -s {dir_fullpath}/{job_out} '
                        f'{dir_basename}.{job_out}'
                    )

        self.prerun_cmds.append(
            f'ln -s {dir_fullpath}/{self.roofline_script1}')
        self.prerun_cmds.append(
            f'ln -s {dir_fullpath}/{self.roofline_script2}')
        self.prerun_cmds.append(f'ln -s {dir_fullpath}/Plot/')
    # }}}
# }}}
# }}}


# {{{ Intel Broadwell
# {{{ BWL_RunErt
@rfm.parameterized_test(
    *[
        [ert_precision, ert_flop, ert_mpi_task]
        for ert_precision in ert_precisions
        for ert_flop in ert_flops
        for ert_mpi_task in cpu_specs['BWL']['mpi_tasks']
    ]
)
class BWL_RunErt(RunErt_Base):
    def __init__(self, ert_precision, ert_flop, ert_mpi_task):
        # {{{ pe
        cpu = 'BWL'
        self.descr = f'Collect ERT data from INTEL {cpu}'
        self.valid_systems = ['dom:mc']
        self.valid_prog_environs = ['PrgEnv-gnu']
        # }}}

        # {{{ build
        self.ert_trials_min = 1
        self.ert_precision = ert_precision
        self.ert_flop = ert_flop
        # }}}

        # {{{ run
        self.num_tasks = ert_mpi_task
        self.num_tasks_per_node = ert_mpi_task
        self.num_cpus_per_task = 36 // ert_mpi_task
        # NOTE: mpi*openmp: [1 36| 2 18| 3 12| 4 9| 6 6| 9 4| 12 3| 18 2| 36 1]
        self.num_tasks_per_core = 1
        self.use_multithreading = False
        self.exclusive = True
        # Assuming ert_repeat=2, time can be adjusted as:
        if ert_flop >= 64 or ert_mpi_task >= 12:
            self.time_limit = '20m'
        else:
            self.time_limit = '10m'
        # The other steps are in the base class
        # }}}
# }}}


# {{{ BWL_PlotErt
@rfm.simple_test
class BWL_PlotErt(PlotErt_Base):
    """
    The Empirical Roofline Tool, ERT, automatically generates roofline data.
    https://bitbucket.org/berkeleylab/cs-roofline-toolkit/
    This class depends on the BWL_RunErt class.
    It can be run with: -n BWL_PlotErt -r
    """

    def __init__(self):
        super().__init__()
        cpu = 'BWL'
        self.descr = f'Plot ERT data on the Roofline chart (INTEL {cpu})'
        self.valid_systems = ['dom:login']
        self.valid_prog_environs = ['PrgEnv-gnu']
        self.maintainers = ['JG']
        self.tags = {'cpu'}
        self.sourcesdir = None
        # gnuplot already installed as rpm on dom but keeping as reminder
        # self.modules = ['gnuplot']
        self.dep_name = f'{cpu}_RunErt'
        for ii in ert_precisions:
            for jj in ert_flops:
                for kk in cpu_specs[cpu]['mpi_tasks']:
                    self.depends_on(f'{self.dep_name}_{ii}_{jj}_{kk}',
                                    udeps.by_env)

        # {{{ performance
        self.reference = {
            '*': {
                'gflops': (cpu_specs[cpu]['ref_GFLOPs'], None, None, 'GF/s'),
                'DRAMbw': (cpu_specs[cpu]['ref_DRAMbw'], None, None, 'GB/s'),
            }
        }
        # }}}

    # {{{ hooks
    @rfm.require_deps
    def prepare_logs(self, HWL_RunErt):
        """
        get all the summary files from the compute jobs for postprocessing
        """
        cpu = 'BWL'
        job_out = 'sum'
        for ii in ert_precisions:
            for jj in ert_flops:
                for kk in cpu_specs[cpu]['mpi_tasks']:
                    dir_fullpath = self.getdep(
                        f'{self.dep_name}_{ii}_{jj}_{kk}', part='mc'
                    ).stagedir
                    dir_basename = dir_fullpath.split('/')[-1]
                    self.prerun_cmds.append(
                        f'ln -s {dir_fullpath}/{job_out} '
                        f'{dir_basename}.{job_out}'
                    )

        self.prerun_cmds.append(
            f'ln -s {dir_fullpath}/{self.roofline_script1}')
        self.prerun_cmds.append(
            f'ln -s {dir_fullpath}/{self.roofline_script2}')
        self.prerun_cmds.append(f'ln -s {dir_fullpath}/Plot/')
    # }}}
# }}}
# }}}
