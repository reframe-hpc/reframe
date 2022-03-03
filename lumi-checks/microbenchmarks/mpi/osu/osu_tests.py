import reframe as rfm
import reframe.utility.sanity as sn

from hpctestlib.microbenchmarks.mpi.osu import (osu_latency,
                                                osu_bandwidth,
                                                build_osu_benchmarks)


cpu_build_variant = build_osu_benchmarks.get_variant_nums(
    build_type='cpu'
)


@rfm.simple_test
class alltoall_check(osu_latency):
    ctrl_msg_size = 8
    perf_msg_size = 8
    executable = 'osu_alltoall'
    osu_binaries = fixture(build_osu_benchmarks, scope='environment',
                           variants=cpu_build_variant)
    valid_systems = ['lumi:small']
    valid_prog_environs = ['PrgEnv-gnu', 'PrgEnv-cray']
    strict_check = False
    reference = {
        'lumi:small': {
            'latency': (8.23, None, 0.1, 'us')
        },
    }
    num_tasks_per_node = 1
    num_gpus_per_node  = 1
    num_tasks = 4
    extra_resources = {
        'switches': {
            'num_switches': 1
        }
    }
    tags = {'benchmark'}
    maintainers = ['RS', 'AJ']


class p2p_config_cscs(rfm.RegressionMixin):
    @run_after('init')
    def cscs_config(self):
        self.num_warmup_iters = 100
        self.num_iters = 1000
        self.num_tasks = 2
        self.num_tasks_per_node = 1
        self.valid_systems = ['lumi:small']
        self.valid_prog_environs = ['PrgEnv-gnu', 'PrgEnv-cray']
        self.exclusive_access = True
        self.maintainers = ['RS', 'AJ']
        self.tags = {'benchmark'}
        self.extra_resources = {
            'switches': {
                'num_switches': 1
            }
        }


@rfm.simple_test
class p2p_bandwidth_cpu_test(osu_bandwidth, p2p_config_cscs):
    descr = 'P2P bandwidth microbenchmark'
    ctrl_msg_size = 4194304
    perf_msg_size = 4194304
    executable = 'osu_bw'
    osu_binaries = fixture(build_osu_benchmarks, scope='environment',
                           variants=cpu_build_variant)
    reference = {
        'lumi:small': {
            'bw': (9607.0, -0.10, None, 'MB/s')
        }
    }


@rfm.simple_test
class p2p_latency_cpu_test(osu_latency, p2p_config_cscs):
    descr = 'P2P latency microbenchmark'
    ctrl_msg_size = 8
    perf_msg_size = 8
    executable = 'osu_latency'
    osu_binaries = fixture(build_osu_benchmarks, scope='environment',
                           variants=cpu_build_variant)
    reference = {
        'lumi:small': {
            'latency': (1.30, None, 0.70, 'us')
        },
    }
