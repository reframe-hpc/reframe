import os

import reframe as rfm
import reframe.utility.sanity as sn


@rfm.parameterized_test(
    *[[gpudims, flop, repeat]
      # gpudims sets (gpu_blocks, gpu_threads):
      for gpudims in [(112, 1024), (224, 512), (448, 256), (896, 128),
                      (1792, 64), (3584, 32)]
      for flop in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
      # self.repeat replaces '-DERT_NUM_EXPERIMENTS=2':
      for repeat in [1, 2]])
class ErtP100Test(rfm.RegressionTest):
    """
    The Empirical Roofline Tool, ERT, empirically generates roofline data:
    https://bitbucket.org/berkeleylab/cs-roofline-toolkit/

    This test checks the ERT tool with NVIDIA Tesla P100-PCIE-16GB:
    Device 0: "Tesla P100-PCIE-16GB"
      CUDA Driver Version / Runtime Version     10.1 / 10.0
      CUDA Capability Major/Minor version number:    6.0
      (56) Multiprocessors, ( 64) CUDA Cores/MP:     3584 CUDA Cores
      GPU Max Clock rate:                            1329 MHz (1.33 GHz)
      Theoretical peak performance per GPU:          4761 Gflop/s
      Maximum number of threads per multiprocessor:  2048
      Peak number of threads:                        114688 threads <---------
      Maximum number of threads per block:           1024           <---------
    NVRM version: NVIDIA UNIX x86_64 Kernel Module  418.39

    # The following python code can help for a parameter space study:
    # (use --exec-policy=async)
    max_threads_per_block = 1024
    max_threads = 114688
    gpu_threads = max_threads_per_block * 2
    while gpu_threads > 32:
        gpu_threads = gpu_threads // 2
        gpu_blocks = max_threads // gpu_threads
        nth = gpu_threads * gpu_blocks
        print('{} {} {} {}'.format(gpu_blocks, gpu_threads, nth, max_threads))
    """
    def __init__(self, gpudims, flop, repeat):
        super().__init__()
        max_gpu_blocks = 3584
        max_flops = 1024
        max_repeat = 2
        self.descr = 'Empirical Roofline Toolkit'
        self.valid_systems = ['dom:gpu']
        self.valid_prog_environs = ['PrgEnv-gnu']
        self.modules = ['craype-accel-nvidia60']
        self.sourcesdir = os.path.join(self.current_system.resourcesdir,
                                       'roofline', 'ert')
        # A single input file is required for nvcc to work:
        self.build_system = 'SingleSource'
        self.prebuild_cmd = [
            'cat kernel1.c driver1.c | sed "s-^#if ERT-#ifdef ERT-g" > '
            '_gpu.cu']
        self.sourcepath = '_gpu.cu'
        self.executable = 'ert.exe'
        self.build_system.cppflags = [
            # ERT_FLOPS = -DERT_FLOP !
            '-DERT_FLOP=%s' % str(flop),
            '-DERT_ALIGN=32',
            # 1G = 1024^3 = 1073741824:
            '-DERT_MEMORY_MAX=1073741824',
            # ERT_GPU True:
            '-DERT_GPU',
            '-DERT_TRIALS_MIN=1',
            '-DERT_WORKING_SET_MIN=128',
            # '-x cu' explicitly sets the language (cuda) for the src files.
        ]
        self.build_system.ldflags = ['-O3']
        self.maintainers = ['JG']
        self.tags = {'scs'}
        gpu_blocks = gpudims[0]
        gpu_threads = gpudims[1]
        self.name = 'ertgpu_Run.{}_FLOPS.{}_GPUBlocks.{}_GPUThreads.{}'.format(
            repeat, flop, gpu_blocks, gpu_threads)
        self.exclusive = True
        self.num_tasks = 1
        self.num_tasks_per_node = 1
        self.num_cpus_per_task = 1
        self.num_tasks_per_core = 1
        self.use_multithreading = False
        self.variables = {
            'CRAYPE_LINK_TYPE': 'dynamic',
            'OMP_NUM_THREADS': str(self.num_cpus_per_task)
        }
        self.executable_opts = [str(gpu_blocks), str(gpu_threads)]
        self.rpt = '%s.rpt' % self.executable
        # Reference roofline boundaries for NVIDIA Tesla P100-PCIE-16GB:
        GFLOPs = 4355.0
        L1bw = 1724.0
        # Keeping for future reference:
        # L2bw = 855.0
        # L3bw = 547.0
        DRAMbw = 521.0
        self.roofline_rpt = 'rpt'
        # use the latest job to generate the roofline rpt:
        if (gpu_blocks == max_gpu_blocks and flop == max_flops and
           repeat == max_repeat):
            self.post_run = [
                'cat *_job.out | python2 preprocess.py > pre',
                'python2 maximum.py < pre > max',
                'python2 summary.py < max > sum',
                # give enough time for all the dependent jobs to collect data:
                'sleep 60',
                'cat ../ertgpu_Run*/sum | python2 roofline.py > rpt',
            ]
            self.sanity_patterns = sn.all([
                # --- check data type:
                sn.assert_eq(sn.extractsingle(
                    r'^\s+(?P<prec>\w+) \*\s+buf = \(\w+ \*\)'
                    r'_mm_malloc\(PSIZE, ERT_ALIGN\);', 'driver1.c', 'prec'),
                    'double'),
                # --- check ert's roofline results. Typical output is:
                #   4355.20 GFLOPs EMP
                # META_DATA
                # GPU_BLOCKS     1792
                # FLOPS          1024
                # GPU_THREADS    64
                #
                #   1723.95 L1 EMP
                #    521.29 DRAM EMP
                #
                # check GFLOPS:
                sn.assert_reference(sn.extractsingle(
                    r'(?P<GFLOPs>\d+.\d+)\sGFLOPs EMP', self.roofline_rpt,
                    'GFLOPs', float), GFLOPs, -0.1, 0.5),
                # check L1 bandwidth:
                sn.assert_reference(sn.extractsingle(
                    r'(?P<L1bw>\d+.\d+)\sL1 EMP', self.roofline_rpt,
                    'L1bw', float), L1bw, -0.1, 0.3),
                # check DRAM bandwidth:
                sn.assert_reference(sn.extractsingle(
                    r'(?P<DRAMbw>\d+.\d+) DRAM EMPv', self.roofline_rpt,
                    'DRAMbw', float), DRAMbw, -0.1, 0.3),
            ])
        else:
            self.post_run = [
                'cat *_job.out | python2 preprocess.py > pre',
                'python2 maximum.py < pre > max',
                'python2 summary.py < max > sum',
            ]
            self.sanity_patterns = sn.assert_found('GFLOPs', 'sum')
