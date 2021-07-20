# FIXME: Use tuples as dictionary keys as soon as
# https://github.com/eth-cscs/reframe/issues/2022 is in

import reframe as rfm
import reframe.utility.sanity as sn
import reframe.utility.typecheck as typ
from hpctestlib.apps.amber import AmberBaseCheck

daint_gpu_performance = {
    'Cellulose_production_NVE': (30.0, -0.05, None, 'ns/day'),
    'FactorIX_production_NVE': (134.0, -0.05, None, 'ns/day'),
    'JAC_production_NVE': (388.0, -0.05, None, 'ns/day'),
    'JAC_production_NVE_4fs': (742, -0.05, None, 'ns/day'),
}

REFERENCE_GPU_PERFORMANCE = {
    'daint:gpu': daint_gpu_performance,
    'dom:gpu': daint_gpu_performance
}

daint_mc_performance_small = {
    'Cellulose_production_NVE': (8.0, -0.30, None, 'ns/day'),
    'FactorIX_production_NVE': (34.0, -0.30, None, 'ns/day'),
    'JAC_production_NVE': (90.0, -0.30, None, 'ns/day'),
    'JAC_production_NVE_4fs': (150.0, -0.30, None, 'ns/day'),
}

eiger_mc_performance_small = {
    'Cellulose_production_NVE': (3.2, -0.30, None, 'ns/day'),
    'FactorIX_production_NVE': (7.0, -0.30, None, 'ns/day'),
    'JAC_production_NVE': (30.0, -0.30, None, 'ns/day'),
    'JAC_production_NVE_4fs': (45.0, -0.30, None, 'ns/day'),
}

REFERENCE_CPU_PERFORMANCE_SMALL = {
    'daint:mc': daint_mc_performance_small,
    'dom:mc': daint_mc_performance_small,
    'eiger:mc': eiger_mc_performance_small,
    'pilatus:mc': eiger_mc_performance_small,
}

REFERENCE_CPU_PERFORMANCE_LARGE = {
    'daint:mc': {
        'Cellulose_production_NVE': (10.0, -0.30, None, 'ns/day'),
        'FactorIX_production_NVE': (36.0, -0.30, None, 'ns/day'),
        'JAC_production_NVE': (78.0, -0.30, None, 'ns/day'),
        'JAC_production_NVE_4fs': (135.0, -0.30, None, 'ns/day'),
    },
    'eiger:mc': {
        'Cellulose_production_NVE': (1.3, -0.30, None, 'ns/day'),
        'FactorIX_production_NVE': (3.5, -0.30, None, 'ns/day'),
        'JAC_production_NVE': (17.0, -0.30, None, 'ns/day'),
        'JAC_production_NVE_4fs': (30.5, -0.30, None, 'ns/day'),
    },
}

REFERENCE_ENERGY = {
    # every system has a different reference energy and drift
    'Cellulose_production_NVE': (-443246.0, 5.0E-05),
    'FactorIX_production_NVE': (-234188.0, 1.0E-04),
    'JAC_production_NVE_4fs': (-44810.0, 1.0E-03),
    'JAC_production_NVE': (-58138.0, 5.0E-04),
}


class AmberCheck(AmberBaseCheck):
    benchmark = parameter([
        # NVE simulations
        'Cellulose_production_NVE',
        'FactorIX_production_NVE',
        'JAC_production_NVE_4fs',
        'JAC_production_NVE',
    ])
    strict_check = False
    extra_resources = {
        'switches': {
            'num_switches': 1
        }
    }
    ener_ref = REFERENCE_ENERGY
    output_file = 'amber.out'
    maintainers = ['VH', 'SO']

    @run_after('init')
    def source_install(self):
        self.prerun_cmds = [
            # cannot use wget because it is not installed on eiger
            f'curl -LJO https://github.com/victorusu/amber_benchmark_suite'
            f'/raw/main/amber_16_benchmark_suite/PME/{self.benchmark}.tar.bz2',
            f'tar xf {self.benchmark}.tar.bz2'
        ]


@rfm.simple_test
class AmberGPUCheck(AmberCheck):
    input_file = 'mdin.GPU'
    valid_systems = ['daint:gpu', 'dom:gpu']
    valid_prog_environs = ['builtin']
    executable = 'pmemd.cuda.MPI'
    num_tasks = 1
    num_gpus_per_node = 1
    num_tasks_per_node = 1
    descr = f'Amber GPU check'
    tags = {'maintenance', 'production', 'health'}

    @run_after('setup')
    def set_perf_reference(self):
        self.reference = REFERENCE_GPU_PERFORMANCE


@rfm.simple_test
class AmberCPUCheck(AmberCheck):
    tags = {'maintenance', 'production'}
    scale = parameter(['small', 'large'])
    valid_systems = ['daint:mc', 'eiger:mc']
    valid_prog_environs = ['builtin']
    executable = 'pmemd.MPI'
    input_file = 'mdin.CPU'

    @run_after('init')
    def set_description(self):
        self.mydescr = f'Amber parallel {self.scale} CPU check'

    @run_after('init')
    def set_additional_systems(self):
        if self.scale == 'small':
            self.valid_systems += ['dom:mc', 'pilatus:mc']

    @run_after('init')
    def set_hierarchical_prgenvs(self):
        if self.current_system.name in ['eiger', 'pilatus']:
            self.valid_prog_environs = ['cpeIntel']

    @run_after('setup')
    def set_perf_reference(self):
        if self.scale == 'small':
            self.reference = REFERENCE_CPU_PERFORMANCE_SMALL
        else:
            self.reference = REFERENCE_CPU_PERFORMANCE_LARGE

    @run_after('init')
    def set_num_tasks_cray_xc(self):
        if self.current_system.name in ['daint', 'dom']:
            self.num_tasks_per_node = 36
            if self.scale == 'small':
                self.num_nodes = 6
            else:
                self.num_nodes = 16
            self.num_tasks = self.num_nodes * self.num_tasks_per_node

    @run_after('init')
    def set_num_tasks_cray_shasta(self):
        if self.current_system.name in ['eiger', 'pilatus']:
            self.num_tasks_per_node = 128
            if self.scale == 'small':
                self.num_nodes = 4
            else:
                # there are too many processors, the large jobs cannot start
                # need to decrease to just 8 nodes
                self.num_nodes = 8
            self.num_tasks = self.num_nodes * self.num_tasks_per_node
