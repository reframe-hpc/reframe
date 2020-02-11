import reframe as rfm
import reframe.utility.sanity as sn


class GPPBaseTest(rfm.RegressionTest):
    '''This test checks the values reported by NVIDIA nvprof for roofline
       modeling:
       - https://github.com/cyanguwa/nersc-roofline/tree/master/GPP
            (compile.survey and run.survey)
       - https://cug.org/proceedings/protected/cug2019_proceedings/includes/
            files/pap103s2-file1.pdf
    '''
    def __init__(self):
        super().__init__()
        self.descr = 'Roofline Analysis of the GPP code using NVIDIA nvprof'
        self.sourcesdir = 'https://github.com/cyanguwa/nersc-roofline.git'
        self.build_system = 'Make'
        self.build_system.cxx = 'nvcc'
        self.maintainers = ['JG']
        self.tags = {'scs'}

    @property
    @sn.sanity_function
    def flops(self):
        flop_count_dp_avg = sn.extractsingle(
            r'^.*flop_count_dp\s+Floating Point Operations\(Double Precision\)'
            r'\s+.*(?P<x>\d\.\d+e\+\d+)$', self.stderr, 'x', float)
        # print("#debug: flop_count_dp_avg={}".format(flop_count_dp_avg))
        return flop_count_dp_avg

    @property
    @sn.sanity_function
    def gflops_per_seconds(self):
        sec = sn.extractsingle(
            r'^\*+\sKernel Time Taken\s\*+=\s(?P<sec>\d+.\d+)\ssecs',
            self.stdout, 'sec', float)
        # print("#debug: sec={}".format(sec))
        # print("#debug: flops={}".format(self.flops))
        # print("#debug: gflops_per_seconds={}".format(self.flops/(sec*10**9)))
        return (self.flops / (sec*10**9))

    @property
    @sn.sanity_function
    def hbm_bytes(self):
        dram_read_transactions_avg = sn.extractsingle(
            r'^.*dram_read_transactions\s+Device Memory Read Transactions\s+.*'
            r'(?P<x>\d\.\d+e\+\d+)$', self.stderr, 'x', float)
        dram_write_transactions_avg = sn.extractsingle(
            r'^.*dram_write_transactions\s+Device Memory Write Transactions\s+'
            r'\d+\s+\d+\s+(?P<x>\d+)$', self.stderr, 'x', float)
        transactions_size = 32.0
        bytes = dram_read_transactions_avg + dram_write_transactions_avg
        bytes = bytes * transactions_size
        # print("#debug: dram_read_avg={}".format(dram_read_transactions_avg))
        # print("#debug: dram_wr_avg={}".format(dram_write_transactions_avg))
        # print("#debug: hbm_bytes={}".format(bytes))
        return bytes

    @property
    @sn.sanity_function
    def arithmetic_intensity(self):
        # print("#debug: ai={}".format(self.flops/self.hbm_bytes))
        return (self.flops / self.hbm_bytes)


@rfm.parameterized_test(*[[iw, repeat, cache]
                        for iw in [6]
                        for repeat in [1, 2]
                        for cache in ['HBM']])
# To reproduce published results (on V100):
#                        for iw in [1, 2, 3, 4, 5, 6]
#                        for repeat in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
#                        for cache in ['L1', 'L2', 'HBM']])
class P100Test(GPPBaseTest):
    ''' Counters for Pascal P100 GPU:
        userguide = 'https://docs.nvidia.com/cuda/profiler-users-guide'
        metrics = '%s/index.html#metrics-reference-6x' % userguide
    '''
    def __init__(self, iw, repeat, cache):
        super().__init__()
        self.name = 'roofline_gpp_P100_iw{}_repeat{}_{}cache'.format(
                    iw, repeat, cache)
        self.valid_systems = ['dom:gpu']
        self.valid_prog_environs = ['PrgEnv-gnu']
        self.modules = ['craype-accel-nvidia60']
        self.prebuild_cmd = [
            'cd GPP/Volta',
            # Pascal P100 GPU:
            'sed -i "s-sm_70-sm_60-" Makefile',
            # fma (fmad=true) vs nofma (fmad=false):
            'sed -i "s/fmad=.*/fmad=true/g" Makefile',
            # iw (loop size):
            'sed -i "s/#define nend.*/#define nend %s/g" GPUComplex.h' % iw,
        ]
        self.executable = './fma_iw{}_rep{}_{}.exe'.format(iw, repeat, cache)
        self.build_system.options = ['EXE=../../%s' % self.executable]
        # 1: <n_bands> 2: <n_valence_bands> 3: <n_plane_waves>
        # 4: <nodes_per_mpi_group> 5: <stride>
        self.executable_opts = ['512', '2', '32768', '20', '0']
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
        self.nvprof_metrics = {
            'L1': ['flop_count_dp', 'gld_transactions', 'gst_transactions',
                   'atomic_transactions', 'local_load_transactions',
                   'local_store_transactions', 'shared_load_transactions',
                   'shared_store_transactions'],
            'L2': ['flop_count_dp', 'l2_read_transactions',
                   'l2_write_transactions'],
            'HBM': ['flop_count_dp', 'dram_read_transactions',
                    'dram_write_transactions'],
            'PCIe/NVLINK': ['flop_count_dp', 'system_read_transactions',
                            'system_write_transactions']
        }
        sep = ' --metrics '
        nvmetrics = sep.join(self.nvprof_metrics[cache])
        self.post_run = [
            'nvprof --kernels "NumBandNgpown_kernel" --metrics %s %s %s' %
            (nvmetrics, self.executable, ' '.join(self.executable_opts))
        ]
        # References for Nvidia P100 (HBM, iw=6):
        gflops = 2796.6
        ai = 13.6
        self.sanity_patterns = sn.all([
            sn.assert_found('P100-PCIE-16GB', self.stderr),
            sn.assert_reference(self.gflops_per_seconds, gflops, -0.5, 0.5),
            sn.assert_reference(self.arithmetic_intensity, ai, -0.5, 0.5),
        ])
