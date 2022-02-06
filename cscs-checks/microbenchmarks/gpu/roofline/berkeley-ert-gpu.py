# Copyright 2016-2022 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import os
from shutil import which
from math import log
import reframe as rfm
import reframe.utility.sanity as sn
import reframe.utility.udeps as udeps

# NOTE: do not run this check with --system (because of deps)
ert_precisions = ['ERT_FP64']
repeat = 1
ert_flops = [1]
# ert_flops = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
gpu_specs = {
    'P100': {
        'capability': 'sm_60',
        'multiprocessors': 56,
        'maximum_number_of_threads_per_multiprocessor': 2048,
        'maximum_number_of_threads_per_block': 1024,
        'warp_size': 32,
        # 4360.14 FP64 GFLOPs EMP GFLOP/sec
        # 1721.88 L1 EMP          GB/sec
        #  523.62 DRAM EMP        GB/sec
    },
    'V100': {
        'capability': 'sm_70',
        'multiprocessors': 80,
        'maximum_number_of_threads_per_multiprocessor': 2048,
        'maximum_number_of_threads_per_block': 1024,
        'warp_size': 32,
        # 7818.02 FP64 GFLOPs EMP   GFLOP/sec
        # 2674.54 L1 EMP            GB/sec
        # 2234.38 L2 EMP            GB/sec
        #  556.68 DRAM EMP          GB/sec
    },
}


# {{{ class RunErt_Base
class RunErt_Base(rfm.RegressionTest):
    descr = f'Empirical Roofline Toolkit (Base for building/running)'
    # pe step
    # build step
    # run step
    # postprocess step
    # sanity step

    # {{{ hooks
    @run_before('compile')
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
        self.sourcepath = 'Kernels/kernel1.cxx Drivers/driver1.cxx'
        self.build_system = 'SingleSource'
        # get all parameters:
        capability = sn.getattr(self, 'cap')
        ert_trials_min = sn.getattr(self, 'ert_trials_min')
        ert_precision = sn.getattr(self, 'ert_precision')
        ert_flop = sn.getattr(self, 'ert_flop')
        self.build_system.cppflags = [
            '-I./Kernels',
            '-DERT_ALIGN=32',
            '-DERT_MEMORY_MAX=1073741824',
            '-DERT_WORKING_SET_MIN=128',
            '-DERT_WSS_MULT=1.1',
            '-DERT_GPU',
            f'-DERT_TRIALS_MIN={ert_trials_min}',
            f'-D{ert_precision}',
            f'-DERT_FLOP={ert_flop}',
        ]
        self.prgenv_flags = {
            'PrgEnv-gnu': ['-O3', '-x cu', f'-arch={capability}'],
        }
        self.build_system.cxx = 'nvcc'
        self.build_system.cxxflags = \
            self.prgenv_flags[self.current_environ.name]
        self.prebuild_cmds = ['module list', 'which gcc', 'which nvcc']

    @run_before('run')
    def set_run_cmds(self):
        # Usage: ./exe gpu_blocks gpu_threads
        ert_gpu_blocks = sn.getattr(self, 'ert_gpu_blocks')
        ert_gpu_threads = sn.getattr(self, 'ert_gpu_threads')
        self.prerun_cmds += [f'for ii in `seq {repeat}`;do']
        self.executable_opts = [
            f'{ert_gpu_blocks} {ert_gpu_threads} &> try.00$ii'
        ]
        self.postrun_cmds += [
            'done',
            'cat try.00* | ./Scripts/preprocess.py > pre',
            './Scripts/maximum.py < pre > max',
            './Scripts/summary.py < max > sum',
        ]

    @sanity_function
    def set_sanity(self):
        return sn.all(
            [
                sn.assert_found(r'^fp64', 'try.001'),
                sn.assert_found(r'^fp64', 'max'),
                sn.assert_found(r'GFLOPs|DRAM', 'sum'),
                sn.assert_found('META_DATA', 'sum'),
            ]
        )
    # }}}
# }}}


# {{{ class PlotErt_Base
class PlotErt_Base(rfm.RunOnlyRegressionTest):
    descr = f'Empirical Roofline Toolkit (Base for plotting)'
    roofline_script1_fname = 'roofline.py'
    roofline_script1 = f'./Scripts/{roofline_script1_fname}'
    roofline_script2 = './ert_cscs.py'
    roofline_out_script1 = 'o.roofline'
    roofline_summary = 'sum'
    executable = f'cat'
    executable_opts = [
        r'*.sum',
        r'|',
        'python3',
        roofline_script1_fname,
        '&> ',
        roofline_out_script1,
    ]

    @sanity_function
    def set_sanity(self):
        return sn.all(
            [
                sn.assert_found(r'GFLOPs EMP', self.roofline_out_script1),
                sn.assert_found(r'DRAM EMP', self.roofline_out_script1),
                sn.assert_found('Empirical roofline graph:', self.stdout),
            ]
        )

    @performance_function('GF/s')
    def gflops(self):
        regex_gflops = r'(\S+)\sFP64 GFLOPs EMP'
        return sn.extractsingle(regex_gflops, self.roofline_out_script1, 1,
                                float)

    @performance_function('GB/s')
    def DRAMbw(self):
        regex_DRAMbw = r'(\S+)\sDRAM EMP'
        return sn.extractsingle(regex_DRAMbw, self.roofline_out_script1, 1,
                                float)

    # {{{ hooks
    @run_before('run')
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


# {{{ P100_RunErt
@rfm.simple_test
class P100_RunErt(RunErt_Base):
    ert_precision = parameter(ert_precisions)
    ert_flop = parameter(ert_flops)
    ert_gpu_threads = parameter([int(2 ** (log(32, 2)))])
    # {{{ pe
    gpu = 'V100'
    descr = f'Collect ERT data from NVIDIA {gpu}'
    valid_systems = ['dom:gpu']
    valid_prog_environs = ['PrgEnv-gnu']
    modules = ['craype-accel-nvidia60', 'cdt-cuda']
    # }}}

    # {{{ build
    cap = gpu_specs[gpu]['capability']
    ert_trials_min = 1
    # }}}

    # {{{ run
    num_tasks = 1
    num_tasks_per_node = 1
    num_cpus_per_task = 1
    exclusive = True
    time_limit = '10m'
    # set blocks and threads per block:
    maximum_number_of_threads = (
        gpu_specs[gpu]['multiprocessors'] *
        gpu_specs[gpu]['maximum_number_of_threads_per_multiprocessor']
    )

    @run_after('init')
    def set_gpu_blocks(self):
        self.ert_gpu_blocks = int(
            self.maximum_number_of_threads / self.ert_gpu_threads
        )
        # The other steps are in the base class
# }}}
# }}}


# {{{ P100_PlotErt
@rfm.simple_test
class P100_PlotErt(PlotErt_Base):
    """
    The Empirical Roofline Tool, ERT, automatically generates roofline data.
    https://bitbucket.org/berkeleylab/cs-roofline-toolkit/
    This class depends on the HWL_Ert_BaseCheck class.
    """

    gpu = 'P100'
    descr = f'Plot ERT data on the Roofline chart (NVIDIA {gpu})'
    valid_systems = ['dom:login']
    valid_prog_environs = ['PrgEnv-gnu']
    maintainers = ['JG']
    tags = {'gpu'}
    sourcesdir = None
    # gnuplot already installed as rpm on dom but keeping as reminder
    # self.modules = ['gnuplot']
    ert_gpu_threads = [32]
    # uncomment for full search:
    # self.ert_gpu_threads = [2 ** x for x in range(
    #     int(log(gpu_specs[gpu]['warp_size'], 2)),
    #     int(log(gpu_specs[gpu]['maximum_number_of_threads_per_block'],
    #             2) + 1),
    #     )
    # ]
    ref_GFLOPs = 330.0
    ref_DRAMbw = 53.0
    reference = {
        '*': {
            'gflops': (ref_GFLOPs, None, None, 'GF/s'),
            'DRAMbw': (ref_DRAMbw, None, None, 'GB/s'),
        }
    }

    @run_after('init')
    def set_dependencies(self):
        self.dep_name = f'{self.gpu}_RunErt'
        for ii in ert_precisions:
            for jj in ert_flops:
                for kk in self.ert_gpu_threads:
                    self.depends_on(f'{self.dep_name}_{ii}_{jj}_{kk}',
                                    udeps.by_env)

    # {{{ hooks
    @require_deps
    def prepare_logs(self, P100_RunErt):
        """
        get all the summary files from the compute jobs for postprocessing
        """
        job_out = 'sum'
        for ii in ert_precisions:
            for jj in ert_flops:
                for kk in self.ert_gpu_threads:
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


# {{{ V100_RunErt
@rfm.simple_test
class V100_RunErt(RunErt_Base):
    ert_precision = parameter(ert_precisions)
    ert_flop = parameter(ert_flops)
    ert_gpu_threads = parameter([32])
    # {{{ pe
    gpu = 'V100'
    descr = f'Collect ERT data from NVIDIA {gpu}'
    valid_systems = ['tsa:cn']
    valid_prog_environs = ['PrgEnv-gnu']
    # }}}

    # {{{ build
    cap = gpu_specs[gpu]['capability']
    ert_trials_min = 1
    # }}}

    # {{{ run
    num_tasks = 1
    num_tasks_per_node = 1
    num_cpus_per_task = 1
    exclusive = True
    maximum_number_of_threads = (
        gpu_specs[gpu]['multiprocessors'] *
        gpu_specs[gpu]['maximum_number_of_threads_per_multiprocessor']
    )
    time_limit = '5m'

    @run_after('init')
    def set_gpu_blocks(self):
        # set blocks and threads per block:
        self.ert_gpu_blocks = int(
            self.maximum_number_of_threads / self.ert_gpu_threads
        )
        # The other steps are in the base class
    # }}}
    # }}}


# {{{ V100_PlotErt
@rfm.simple_test
class V100_PlotErt(PlotErt_Base):
    """
    The Empirical Roofline Tool, ERT, automatically generates roofline data.
    https://bitbucket.org/berkeleylab/cs-roofline-toolkit/
    This class depends on the HWL_Ert_BaseCheck class.
    """

    gpu = 'V100'
    descr = f'Plot ERT data on the Roofline chart (NVIDIA {gpu})'
    valid_systems = ['tsa:login']
    valid_prog_environs = ['PrgEnv-gnu']
    maintainers = ['JG']
    tags = {'gpu'}
    sourcesdir = None
    modules = ['gnuplot']
    ert_gpu_threads = [32]
    # uncomment for full search:
    # self.ert_gpu_threads = [2 ** x for x in range(
    #     int(log(gpu_specs[gpu]['warp_size'], 2)),
    #     int(log(gpu_specs[gpu]['maximum_number_of_threads_per_block'],
    #             2) + 1),
    #     )
    # ]
    ref_GFLOPs = 330.0
    ref_DRAMbw = 53.0
    reference = {
        '*': {
            'gflops': (ref_GFLOPs, None, None, 'GF/s'),
            'DRAMbw': (ref_DRAMbw, None, None, 'GB/s'),
        }
    }

    @run_after('init')
    def set_dependencies(self):
        self.dep_name = f'{self.gpu}_RunErt'
        for ii in ert_precisions:
            for jj in ert_flops:
                for kk in self.ert_gpu_threads:
                    self.depends_on(f'{self.dep_name}_{ii}_{jj}_{kk}',
                                    udeps.by_env)

    # {{{ hooks
    @require_deps
    def prepare_logs(self, V100_RunErt):
        """
        get all the summary files from the compute jobs for postprocessing
        """
        job_out = 'sum'
        for ii in ert_precisions:
            for jj in ert_flops:
                for kk in self.ert_gpu_threads:
                    dir_fullpath = self.getdep(
                        f'{self.dep_name}_{ii}_{jj}_{kk}', part='cn'
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
