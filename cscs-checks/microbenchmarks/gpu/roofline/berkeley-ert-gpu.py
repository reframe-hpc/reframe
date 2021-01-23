# Copyright 2016-2021 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
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
ert_precisions = ["ERT_FP64"]
repeat = 1
ert_flops = [1]
# ert_flops = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
gpu_specs = {
    "P100": {
        "capability": "sm_60",
        "multiprocessors": 56,
        "maximum_number_of_threads_per_multiprocessor": 2048,
        "maximum_number_of_threads_per_block": 1024,
        "warp_size": 32,
        # 4360.14 FP64 GFLOPs EMP GFLOP/sec
        # 1721.88 L1 EMP          GB/sec
        #  523.62 DRAM EMP        GB/sec
    },
    "V100": {
        "capability": "sm_70",
        "multiprocessors": 80,
        "maximum_number_of_threads_per_multiprocessor": 2048,
        "maximum_number_of_threads_per_block": 1024,
        "warp_size": 32,
        # 7818.02 FP64 GFLOPs EMP   GFLOP/sec
        # 2674.54 L1 EMP            GB/sec
        # 2234.38 L2 EMP            GB/sec
        #  556.68 DRAM EMP          GB/sec
    },
}


# {{{ class RunErt_Base
class RunErt_Base(rfm.RegressionTest):
    def __init__(self):
        self.descr = f"Empirical Roofline Toolkit (Base for building/running)"
        # pe step
        # build step
        # run step
        # postprocess step
        # sanity step

    # {{{ hooks
    @rfm.run_before("compile")
    def set_compiler_flags_and_variables(self):
        self.sourcesdir = os.path.join(
            self.current_system.resourcesdir,
            "roofline",
            "cs-roofline-toolkit.git",
            "Empirical_Roofline_Tool-1.1.0",
        )
        self.readonly_files = [
            "Batch",
            "Config",
            "Drivers",
            "ert",
            "ert_cscs.py",
            "ERT_Users_Manual.pdf",
            "Kernels",
            "Plot",
            "Python",
            "README.md",
            "Results",
            "Scripts",
        ]
        self.sourcepath = "Kernels/kernel1.cxx Drivers/driver1.cxx"
        self.build_system = "SingleSource"
        # get all parameters:
        capability = sn.getattr(self, "cap")
        ert_trials_min = sn.getattr(self, "ert_trials_min")
        ert_precision = sn.getattr(self, "ert_precision")
        ert_flop = sn.getattr(self, "ert_flop")
        self.build_system.cppflags = [
            "-I./Kernels",
            "-DERT_ALIGN=32",
            "-DERT_MEMORY_MAX=1073741824",
            "-DERT_WORKING_SET_MIN=128",
            "-DERT_WSS_MULT=1.1",
            "-DERT_GPU",
            f"-DERT_TRIALS_MIN={ert_trials_min}",
            f"-D{ert_precision}",
            f"-DERT_FLOP={ert_flop}",
        ]
        self.prgenv_flags = {
            "PrgEnv-gnu": ["-O3", "-x cu", f"-arch={capability}"],
        }
        self.build_system.cxx = "nvcc"
        self.build_system.cxxflags = \
            self.prgenv_flags[self.current_environ.name]
        self.prebuild_cmds = ["module list", "which gcc", "which nvcc"]

    @rfm.run_before("run")
    def set_run_cmds(self):
        # Usage: ./exe gpu_blocks gpu_threads
        ert_gpu_blocks = sn.getattr(self, "ert_gpu_blocks")
        ert_gpu_threads = sn.getattr(self, "ert_gpu_threads")
        self.prerun_cmds += [f"for ii in `seq {repeat}`;do"]
        self.executable_opts = [
            f"{ert_gpu_blocks} {ert_gpu_threads} &> try.00$ii"
        ]
        self.postrun_cmds += [
            "done",
            "cat try.00* | ./Scripts/preprocess.py > pre",
            "./Scripts/maximum.py < pre > max",
            "./Scripts/summary.py < max > sum",
        ]

    @rfm.run_before("sanity")
    def set_sanity(self):
        self.sanity_patterns = sn.all(
            [
                sn.assert_found(r"^fp64", "try.001"),
                sn.assert_found(r"^fp64", "max"),
                sn.assert_found(r"GFLOPs|DRAM", "sum"),
                sn.assert_found("META_DATA", "sum"),
            ]
        )
    # }}}
# }}}


# {{{ class PlotErt_Base
class PlotErt_Base(rfm.RunOnlyRegressionTest):
    def __init__(self):
        self.descr = f"Empirical Roofline Toolkit (Base for plotting)"
        self.roofline_script1_fname = "roofline.py"
        self.roofline_script1 = f"./Scripts/{self.roofline_script1_fname}"
        self.roofline_script2 = "./ert_cscs.py"
        self.roofline_out_script1 = "o.roofline"
        self.roofline_summary = "sum"
        self.executable = f"cat"
        self.executable_opts = [
            r"*.sum",
            r"|",
            "python3",
            self.roofline_script1_fname,
            "&> ",
            self.roofline_out_script1,
        ]

        # {{{ sanity_patterns
        self.sanity_patterns = sn.all(
            [
                sn.assert_found(r"GFLOPs EMP", self.roofline_out_script1),
                sn.assert_found(r"DRAM EMP", self.roofline_out_script1),
                sn.assert_found("Empirical roofline graph:", self.stdout),
            ]
        )
        # }}}

        # {{{ performance
        regex_gflops = r"(\S+)\sFP64 GFLOPs EMP"
        regex_L1bw = r"(\S+)\sL1 EMP"
        regex_DRAMbw = r"(\S+)\sDRAM EMP"
        gflops = sn.extractsingle(regex_gflops, self.roofline_out_script1, 1,
                                  float)
        DRAMbw = sn.extractsingle(regex_DRAMbw, self.roofline_out_script1, 1,
                                  float)
        self.perf_patterns = {
            "gflops": gflops,
            "DRAMbw": DRAMbw,
        }
        # }}}

    # {{{ hooks
    @rfm.run_before("run")
    def check_gnuplot(self):
        gnuplot = which("gnuplot")
        if gnuplot is None:
            self.postrun_cmds = [
                self.roofline_script2,
                "# gnuplot roofline.gnuplot",  # name hardcoded in the script
                "# file roofline.ps",  # available formats: gnuplot, json, tex
            ]
        else:
            self.postrun_cmds = [
                self.roofline_script2,
                "gnuplot roofline.gnuplot",
                "file roofline.ps",
            ]
    # }}}
# }}}


# {{{ P100_RunErt
@rfm.parameterized_test(
    *[
        [ert_precision, ert_flop, ert_gpu_threads]
        for ert_precision in ert_precisions
        for ert_flop in ert_flops
        for ert_gpu_threads in [int(2 ** (log(32, 2))]  # = 32
        # uncomment for full search:
        # for ert_gpu_threads in [2 ** x for x in range(
        #     int(log(gpu_specs["P100"]["warp_size"], 2)),
        #     int(log(gpu_specs["P100"]["maximum_number_of_threads_per_block"],
        #             2) + 1),
        #     )
        # ]
    ]
)
class P100_RunErt(RunErt_Base):
    def __init__(self, ert_precision, ert_flop, ert_gpu_threads):
        # {{{ pe
        gpu = "V100"
        self.descr = f"Collect ERT data from NVIDIA {gpu}"
        self.valid_systems = ["dom:gpu"]
        self.valid_prog_environs = ["PrgEnv-gnu"]
        self.modules = ["craype-accel-nvidia60", "cdt-cuda"]
        # }}}

        # {{{ build
        self.cap = gpu_specs[gpu]["capability"]
        self.ert_trials_min = 1
        self.ert_precision = ert_precision
        self.ert_flop = ert_flop
        # }}}

        # {{{ run
        self.num_tasks = 1
        self.num_tasks_per_node = 1
        self.num_cpus_per_task = 1
        self.exclusive = True
        self.time_limit = "10m"
        # set blocks and threads per block:
        self.ert_gpu_threads = ert_gpu_threads
        maximum_number_of_threads = (gpu_specs[gpu]["multiprocessors"] *
            gpu_specs[gpu]["maximum_number_of_threads_per_multiprocessor"]
        )
        self.ert_gpu_blocks = int(maximum_number_of_threads / ert_gpu_threads)
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

    def __init__(self):
        super().__init__()
        gpu = "P100"
        self.descr = f"Plot ERT data on the Roofline chart (NVIDIA {gpu})"
        self.valid_systems = ["dom:login"]
        self.valid_prog_environs = ["PrgEnv-gnu"]
        self.maintainers = ["JG"]
        self.tags = {"gpu"}
        self.sourcesdir = None
        # self.modules = ['gnuplot']
        self.ert_gpu_threads = [32]
        # uncomment for full search:
        # self.ert_gpu_threads = [2 ** x for x in range(
        #     int(log(gpu_specs[gpu]["warp_size"], 2)),
        #     int(log(gpu_specs[gpu]["maximum_number_of_threads_per_block"],
        #             2) + 1),
        #     )
        # ]
        self.dep_name = f"{gpu}_RunErt"
        for ii in ert_precisions:
            for jj in ert_flops:
                for kk in self.ert_gpu_threads:
                    self.depends_on(f"{self.dep_name}_{ii}_{jj}_{kk}",
                                    udeps.by_env)

        # {{{ performance
        # Reference roofline boundaries for NVIDIA P100
        ref_GFLOPs = 330.0
        ref_DRAMbw = 53.0
        self.reference = {
            "*": {
                "gflops": (ref_GFLOPs, None, None, "GF/s"),
                "DRAMbw": (ref_DRAMbw, None, None, "GB/s"),
            }
        }
        # }}}

    # {{{ hooks
    @rfm.require_deps
    def prepare_logs(self, P100_RunErt):
        """
        get all the summary files from the compute jobs for postprocessing
        """
        job_out = "sum"
        for ii in ert_precisions:
            for jj in ert_flops:
                for kk in self.ert_gpu_threads:
                    dir_fullpath = self.getdep(
                        f"{self.dep_name}_{ii}_{jj}_{kk}", part="gpu"
                    ).stagedir
                    dir_basename = dir_fullpath.split("/")[-1]
                    self.prerun_cmds.append(
                        f"ln -s {dir_fullpath}/{job_out} "
                        f"{dir_basename}.{job_out}"
                    )

        self.prerun_cmds.append(
            f"ln -s {dir_fullpath}/{self.roofline_script1}")
        self.prerun_cmds.append(
            f"ln -s {dir_fullpath}/{self.roofline_script2}")
        self.prerun_cmds.append(f"ln -s {dir_fullpath}/Plot/")
    # }}}
# }}}


# {{{ V100_RunErt
@rfm.parameterized_test(
    *[
        [ert_precision, ert_flop, ert_gpu_threads]
        for ert_precision in ert_precisions
        for ert_flop in ert_flops
        for ert_gpu_threads in [32]
        # uncomment for full search:
        # for ert_gpu_threads in [2 ** x for x in range(
        #     int(log(gpu_specs["V100"]["warp_size"], 2)),
        #     int(log(gpu_specs["V100"]["maximum_number_of_threads_per_block"],
        #             2) + 1),
        #     )
        # ]
    ]
)
class V100_RunErt(RunErt_Base):
    def __init__(self, ert_precision, ert_flop, ert_gpu_threads):
        # {{{ pe
        gpu = "V100"
        self.descr = f"Collect ERT data from NVIDIA {gpu}"
        self.valid_systems = ["tsa:cn"]
        self.valid_prog_environs = ["PrgEnv-gnu"]
        # }}}

        # {{{ build
        self.cap = gpu_specs[gpu]["capability"]
        self.ert_trials_min = 1
        self.ert_precision = ert_precision
        self.ert_flop = ert_flop
        # }}}

        # {{{ run
        self.num_tasks = 1
        self.num_tasks_per_node = 1
        self.num_cpus_per_task = 1
        self.exclusive = True
        self.time_limit = "5m"
        # set blocks and threads per block:
        self.ert_gpu_threads = ert_gpu_threads
        maximum_number_of_threads = (gpu_specs[gpu]["multiprocessors"] *
            gpu_specs[gpu]["maximum_number_of_threads_per_multiprocessor"]
        )
        self.ert_gpu_blocks = int(maximum_number_of_threads / ert_gpu_threads)
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

    def __init__(self):
        super().__init__()
        gpu = "V100"
        self.descr = f"Plot ERT data on the Roofline chart (NVIDIA {gpu})"
        self.valid_systems = ["tsa:login"]
        self.valid_prog_environs = ["PrgEnv-gnu"]
        self.maintainers = ["JG"]
        self.tags = {"gpu"}
        self.sourcesdir = None
        self.modules = ["gnuplot"]
        self.ert_gpu_threads = [32]
        # uncomment for full search:
        # self.ert_gpu_threads = [2 ** x for x in range(
        #     int(log(gpu_specs[gpu]["warp_size"], 2)),
        #     int(log(gpu_specs[gpu]["maximum_number_of_threads_per_block"],
        #             2) + 1),
        #     )
        # ]
        self.dep_name = f"{gpu}_RunErt"
        for ii in ert_precisions:
            for jj in ert_flops:
                for kk in self.ert_gpu_threads:
                    self.depends_on(f"{self.dep_name}_{ii}_{jj}_{kk}",
                                    udeps.by_env)

        # {{{ performance
        # Reference roofline boundaries for NVIDIA P100
        ref_GFLOPs = 330.0
        ref_DRAMbw = 53.0
        self.reference = {
            "*": {
                "gflops": (ref_GFLOPs, None, None, "GF/s"),
                "DRAMbw": (ref_DRAMbw, None, None, "GB/s"),
            }
        }
        # }}}

    # {{{ hooks
    @rfm.require_deps
    def prepare_logs(self, V100_RunErt):
        """
        get all the summary files from the compute jobs for postprocessing
        """
        job_out = "sum"
        for ii in ert_precisions:
            for jj in ert_flops:
                for kk in self.ert_gpu_threads:
                    dir_fullpath = self.getdep(
                        f"{self.dep_name}_{ii}_{jj}_{kk}", part="cn"
                    ).stagedir
                    dir_basename = dir_fullpath.split("/")[-1]
                    self.prerun_cmds.append(
                        f"ln -s {dir_fullpath}/{job_out} "
                        f"{dir_basename}.{job_out}"
                    )

        self.prerun_cmds.append(
            f"ln -s {dir_fullpath}/{self.roofline_script1}")
        self.prerun_cmds.append(
            f"ln -s {dir_fullpath}/{self.roofline_script2}")
        self.prerun_cmds.append(f"ln -s {dir_fullpath}/Plot/")
    # }}}
# }}}
