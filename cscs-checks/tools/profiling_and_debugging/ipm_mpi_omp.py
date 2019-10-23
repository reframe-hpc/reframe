import os

import reframe as rfm
import reframe.utility.sanity as sn


@rfm.required_version('>=2.14')
@rfm.parameterized_test(['C++'], ['F90'])
class Ipm(rfm.RegressionTest):
    def __init__(self, lang):
        self.name = 'Ipm_%s' % lang.replace('+', 'p')
        self.descr = self.name
        self.valid_systems = ['daint:gpu', 'daint:mc', 'dom:gpu', 'dom:mc']
        self.valid_prog_environs = ['PrgEnv-cray', 'PrgEnv-cray_classic',
                                    'PrgEnv-gnu', 'PrgEnv-intel', 'PrgEnv-pgi']
        self.prgenv_flags = {
            'PrgEnv-gnu': ['-O2', '-g', '-fopenmp'],
            'PrgEnv-cray': ['-O2', '-g',
                            '-homp' if lang == 'F90' else '-fopenmp'],
            'PrgEnv-cray_classic': ['-O2', '-g', '-homp'],
            'PrgEnv-intel': ['-O2', '-g', '-openmp'],
            'PrgEnv-pgi': ['-O2', '-g', '-mp']
        }
        ipm_ver = '2.0.6'
        tc_ver = '19.09'
        self.ipm_modules = {
            'PrgEnv-gnu': ['IPM/%s-CrayGNU-%s' % (ipm_ver, tc_ver)],
            'PrgEnv-cray': ['IPM/%s-CrayCCE-%s' % (ipm_ver, tc_ver)],
            'PrgEnv-cray_classic': [
                'IPM/%s-CrayCCE-%s-classic' % (ipm_ver, tc_ver)],
            'PrgEnv-intel': ['IPM/%s-CrayIntel-%s' % (ipm_ver, tc_ver)],
            'PrgEnv-pgi': ['IPM/%s-CrayPGI-%s' % (ipm_ver, tc_ver)]
        }
        self.sourcesdir = os.path.join('src', lang)
        self.executable = './jacobi'
        self.build_system = 'Make'

        self.num_tasks = 3
        self.num_tasks_per_node = 3
        self.num_cpus_per_task = 4
        self.num_iterations = 100
        self.variables = {
            'OMP_NUM_THREADS': str(self.num_cpus_per_task),
            'ITERATIONS': str(self.num_iterations),
            'OMP_PROC_BIND': 'true',
            'CRAYPE_LINK_TYPE': 'dynamic',
            'PKG_CONFIG_PATH':
                '$PAT_BUILD_PAPI_LIBDIR/pkgconfig:$PKG_CONFIG_PATH',
            'LD_LIBRARY_PATH': '$PAT_BUILD_PAPI_LIBDIR:'
                               '$LD_LIBRARY_PATH',
            # The list of available hardware performance counters depends
            # on the cpu type:
            #    srun -n1 -t1 -Cgpu papi_avail
            # More infos: http://ipm-hpc.sourceforge.net/userguide.html
            'IPM_HPM': 'PAPI_L1_TCM,PAPI_L2_TCM,PAPI_L3_TCM',
        }
        self.txtrpt = 'ipm.rpt'
        self.post_run = ['ipm_parse.pl -h',
                         'ipm_parse.pl -full *.ipm.xml &> %s' % self.txtrpt,
                         'ipm_parse.pl -html *.ipm.xml',
                         'cp *ipm.xml_ipm*/index.html .']
        self.maintainers = ['JG']
        self.tags = {'production'}

    def setup(self, partition, environ, **job_opts):
        self.modules = self.ipm_modules[environ.name]
        super().setup(partition, environ, **job_opts)
        environ_name = self.current_environ.name
        prgenv_flags = self.prgenv_flags[environ_name]
        self.build_system.cflags = prgenv_flags
        self.build_system.cxxflags = prgenv_flags
        self.build_system.fflags = prgenv_flags
        self.build_system.ldflags = ['-lm', '`pkg-config --libs papi`',
                                     '`pkg-config --libs pfm`', '${IPM}']
        self.htmlrpt = 'index.html'
        self.sanity_patterns = sn.all([
            # check the job:
            sn.assert_found('SUCCESS', self.stdout),
            # check the txt report:
            sn.assert_reference(sn.extractsingle(
                r'^#\sPAPI_L1_TCM\s+(?P<totalmissesL1>\S\.\S+)',
                self.txtrpt, 'totalmissesL1', float), 91159658, -0.1, 0.1),
            # check the html report:
            sn.assert_reference(sn.extractsingle(
                r'^<tr><td>\sPAPI_L1_TCM\s<\/td><td\salign=right>\s'
                r'(?P<totalmissesL1>\d+)',
                self.htmlrpt, 'totalmissesL1', float), 91159658, -0.1, 0.1),
        ])
