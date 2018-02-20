import os
import itertools
import reframe.utility.sanity as sn

from reframe.core.pipeline import RegressionTest


class G2GMeteoswissTest(RegressionTest):
    def __init__(self, g2g, **kwargs):
        super().__init__('g2g_%s_meteoswiss_check' % g2g,
                         os.path.dirname(__file__), **kwargs)

        self.descr = 'G2G Meteoswiss check with G2G=%s' % g2g
        self.strict_check = False
        self.valid_systems = ['kesch:cn']
        self.valid_prog_environs = ['PrgEnv-gdr']
        self.executable = 'src/$EXECUTABLE'
        self.sourcesdir = ('https://github.com/MeteoSwiss-APN/'
                           'comm_overlap_bench.git')
        self.sourcepath = 'src'

        self.maintainers = ['TM', 'JG']
        self.tags = {'production'}

        self.num_tasks = 2
        self.num_gpus_per_node  = 2

        cuda_visible_devices = {1: r'CUDA_VISIBLE_DEVICES: '
                                   r'\[0: \d\] \[1: \d\]',
                                2: r'CUDA_VISIBLE_DEVICES: '
                                   r'\[0: \d,\d\] \[1: \d,\d\]'}

        self.sanity_patterns = sn.all([
            sn.assert_found('ELAPSED TIME:', self.stdout),
            sn.assert_found(cuda_visible_devices[g2g], self.stdout)
        ])

        self.perf_patterns = {
            'perf': sn.extractsingle(r'ELAPSED TIME:\s+(?P<perf>\S+)',
                                     self.stdout, 'perf', float)
        }

        self.reference = {
            'kesch:cn': {'perf': (3.00, None, 0.2)}
        }

        self.variables = {'G2G': str(g2g),
                          'LD_PRELOAD': '$(pkg-config --variable=libdir '
                                        'mvapich2-gdr)/libmpi.so'}

        self.prebuild_cmd = ['git checkout barebones']

    def setup(self, partition, environ, **job_opts):
        super().setup(partition, environ, **job_opts)
        self.job.pre_run = ["export EXECUTABLE=$(ls %s/src/ | "
                            "grep 'GNU.*MVAPICH.*CUDA.*kesch.*')"
                            % self.stagedir]

    def compile(self):
        super().compile(makefile='../makefiles/makefile-kesch',
                        options='NVCC=nvcc')


def _get_checks(**kwargs):
    return [G2GMeteoswissTest(1, **kwargs),
            G2GMeteoswissTest(2, **kwargs)]
