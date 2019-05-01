import os

import reframe as rfm
import reframe.utility.sanity as sn


@rfm.parameterized_test(*[[repeat, toolsversion, datalayout]
                          for repeat in ['100000']
                          for toolsversion in ['8.35.0']
                          for datalayout in ['G3_AOS_SCALAR', 'G3_SOA_SCALAR',
                                             'G3_AOS_VECTOR', 'G3_SOA_VECTOR']
                          ])
class IntelRooflineSdeTest(rfm.RegressionTest):
    '''This test checks the values reported by Intel SDE for roofline modeling:
       - https://software.intel.com/en-us/articles/
            intel-software-development-emulator
       - https://bitbucket.org/dwdoerf/stream-ai-example/src/master/
       - https://www.nersc.gov/
            users/application-performance/measuring-arithmetic-intensity
    '''
    def __init__(self, repeat, toolsversion, datalayout):
        super().__init__()
        self.descr = 'Roofline Analysis test with Intel SDE'
        self.valid_systems = ['dom:mc']
        # Reporting MFLOPS is not available on Intel Haswell cpus, see
        # https://www.intel.fr/content/dam/www/public/us/en/documents/manuals/
        # 64-ia-32-architectures-software-developer-vol-1-manual.pdf
        self.valid_prog_environs = ['PrgEnv-intel']
        self.sourcesdir = os.path.join(self.current_system.resourcesdir,
                                       'roofline', 'intel_advisor')
        self.build_system = 'SingleSource'
        self.sourcepath = '_roofline.cpp'
        self.prebuild_cmd = [
            'patch < SDE/roofline_template.patch',
            'sed -e "s-XXXX-%s-" -e "s-YYYY-%s-" %s &> %s' %
            (repeat, datalayout, 'roofline_template.cpp', '_roofline.cpp')
        ]
        self.build_system.cppflags = ['-D_SDE']
        self.build_system.ldflags = ['-g', '-O3', '-qopenmp', '-restrict',
                                     '-qopt-streaming-stores', 'always',
                                     '-std=c++11']
        self.exclusive = True
        self.num_tasks = 1
        self.num_tasks_per_node = 1
        self.num_cpus_per_task = 1
        self.num_tasks_per_core = 1
        self.use_multithreading = False
        self.variables = {
            'OMP_NUM_THREADS': str(self.num_cpus_per_task),
            'CRAYPE_LINK_TYPE': 'dynamic',
        }
        self.executable = 'sde'
        self.target_executable = './roof.exe'
        self.sde = '%s.sde' % self.target_executable
        self.rpt = '%s.rpt' % self.target_executable
        self.pre_run = [
            'mv %s %s' % (self.executable, self.target_executable),
            'module use /apps/dom/UES/jenkins/7.0.UP00/mc/easybuild/'
            'experimental/modules/all',
            'module load sde',
            'sde -help'
        ]
        self.sdeflags = ['%s -d -iform 1 -omix %s -i -top_blocks 500 '
                         '-global_region -start_ssc_mark 111:repeat '
                         '-stop_ssc_mark 222:repeat -- %s' %
                         ('-bdw', self.sde, self.target_executable)]
        self.executable_opts = self.sdeflags
        self.sanity_patterns = sn.assert_found('Total FLOPs =', self.rpt)
        self.post_run = ['SDE/parse-sde.sh %s.* &> %s' % (self.sde, self.rpt)]
        self.maintainers = ['JG']
        self.tags = {'production'}
        self.sanity_patterns = sn.all([
            sn.assert_eq(sn.extractsingle(
                r'^Intel\(R\) Software Development Emulator\.  Version:  '
                r'(?P<toolsversion>\d+\.\d+\.\d+)', self.stdout,
                'toolsversion'), toolsversion),
        ])
        # References for Intel Broadwell CPU (E5-2695 v4):
        references = {
            'G3_AOS_SCALAR': {
                'dom:mc': {
                    'gflops': (596, -0.1, 0.3, 'Gflop/s'),
                    'ai': (0.16, -0.05, 0.05, 'flop/byte')
                }
            },
            'G3_SOA_SCALAR': {
                'dom:mc': {
                    'gflops': (612, -0.1, 0.3, 'Gflop/s'),
                    'ai': (0.16, -0.05, 0.05, 'flop/byte')
                }
            },
            'G3_AOS_VECTOR': {
                'dom:mc': {
                    'gflops': (1152, -0.1, 0.3, 'Gflop/s'),
                    'ai': (0.125, -0.05, 0.05, 'flop/byte')
                }
            },
            'G3_SOA_VECTOR': {
                'dom:mc': {
                    'gflops': (1125, -0.1, 0.3, 'Gflop/s'),
                    'ai': (0.16, -0.05, 0.05, 'flop/byte')
                }
            },
        }
        self.reference = references[datalayout]
        self.perf_patterns = {
            'gflops': self.gflops,
            'ai': self.arithmetic_intensity,
        }

    @property
    @sn.sanity_function
    def arithmetic_intensity(self):
        flops = sn.extractsingle(r'^--->Total FLOPs = (?P<flops>\d+)',
                                 self.rpt, 'flops', int)
        byts = sn.extractsingle(r'^--->Total Bytes = (?P<byts>\d+)',
                                self.rpt, 'byts', int)
        # debug: print('ai={}'.format(flops/byts))
        return flops/byts

    @property
    @sn.sanity_function
    def gflops(self):
        flops = sn.extractsingle(r'^--->Total FLOPs = (?P<flops>\d+)',
                                 self.rpt, 'flops', int)
        msec = sn.extractsingle(r'^elapsed time: (?P<msec>\d+)ms', self.stdout,
                                'msec', float)
        # debug: print('gflops={}'.format(flops/((msec/1000)*10**6)))
        return flops/((msec/1000)*10**6)
