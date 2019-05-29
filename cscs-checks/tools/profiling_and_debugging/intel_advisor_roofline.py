import os

import reframe as rfm
import reframe.utility.sanity as sn


@rfm.required_version('>=2.14')
@rfm.parameterized_test(*[[repeat, toolsversion, datalayout]
                          for repeat in ['100000']
                          for toolsversion in ['597843']
                          for datalayout in ['G3_AOS_SCALAR', 'G3_SOA_SCALAR',
                                             'G3_AOS_VECTOR', 'G3_SOA_VECTOR']
                          ])
class IntelRooflineTest(rfm.RegressionTest):
    '''This test checks the values reported by Intel Advisor's roofline model:
    https://software.intel.com/en-us/intel-advisor-xe

    The roofline model is based on GFLOPS and Arithmetic Intensity (AI):
      "Self GFLOPS" = "Self GFLOP" / "Self Elapsed Time"
      "Self GB/s" = "Self Memory GB" / "Self Elapsed Time"
      "Self AI" = "Self GFLOPS" / "Self GB/s"

    While a roofline analysis flag exists ('advixe-cl -collect roofline'), it
    may not be used to collect data on MPI applications; in that case, the
    survey and flops analysis must be collected separately: first run a survey
    analysis ('advixe-cl -collect survey') and then run a tripcounts+flops
    analysis ('advixe-cl -collect tripcounts -flop') using the same project
    directory for both steps.

    Example result on 1 core of Intel Broadwell CPU (E5-2695 v4):
        G3_AOS_SCALAR: self_gflops,  2.79  self_arithmetic_intensity', 0.166
        G3_AOS_VECTOR: self_gflops,  3.79  self_arithmetic_intensity', 0.125
        G3_SOA_SCALAR: self_gflops,  2.79  self_arithmetic_intensity', 0.166
        G3_SOA_VECTOR: self_gflops, 10.62  self_arithmetic_intensity', 0.166
    '''
    def __init__(self, repeat, toolsversion, datalayout):
        super().__init__()
        self.descr = 'Roofline Analysis test with Intel Advisor'
        # advisor/2019 is failing on dom ("Exceeded job memory limit")
        # https://webrt.cscs.ch/Ticket/Display.html?id=36087
        self.valid_systems = ['daint:mc']
        # Reporting MFLOPS is not available on Intel Haswell cpus, see
        # https://www.intel.fr/content/dam/www/public/us/en/documents/manuals/
        # 64-ia-32-architectures-software-developer-vol-1-manual.pdf
        self.valid_prog_environs = ['PrgEnv-intel']
        # Testing with advisor/2018 (build 551025) fails with:
        #    roof.dir/nid00753.000/trc000/trc000.advixe
        #    Application exit code: 139
        self.modules = ['advisor/2019_update4']
        self.sourcesdir = os.path.join(self.current_system.resourcesdir,
                                       'roofline', 'intel_advisor')
        self.build_system = 'SingleSource'
        self.sourcepath = '_roofline.cpp'
        self.executable = 'advixe-cl'
        self.target_executable = './roof.exe'
        self.build_system.cppflags = ['-D_ADVISOR',
                                      '-I$ADVISOR_2019_DIR/include']
        self.prgenv_flags = {
            'PrgEnv-intel': ['-g', '-O2', '-std=c++11', '-restrict'],
        }
        self.build_system.ldflags = ['-L$ADVISOR_2019_DIR/lib64 -littnotify']
        # self.roofline_rpt = 'Intel_Advisor_roofline_results.rpt'
        self.roofline_rpt = '%s.rpt' % self.target_executable
        self.version_rpt = 'Intel_Advisor_version.rpt'
        self.roofline_ref = 'Intel_Advisor_roofline_reference.rpt'
        self.prebuild_cmd = [
            'patch < ADVISOR/roofline_template.patch',
            'sed -e "s-XXXX-%s-" -e "s-YYYY-%s-" %s &> %s' %
            (repeat, datalayout, 'roofline_template.cpp', '_roofline.cpp')
        ]
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
        self.pre_run = [
            'mv %s %s' % (self.executable, self.target_executable),
            'advixe-cl -help collect | head -20',
        ]
        self.roofdir = './roof.dir'
        self.executable_opts = [
            '--collect survey --project-dir=%s --search-dir src:rp=. '
            '--data-limit=0 --no-auto-finalize --trace-mpi -- %s ' %
            (self.roofdir, self.target_executable)
        ]
        # Reference roofline boundaries for Intel Broadwell CPU (E5-2695 v4):
        L1bw = 293*1024**3
        L2bw = 79*1024**3
        L3bw = 33*1024**3
        DPfmabw = 45*1024**3
        DPaddbw = 12*1024**3
        ScalarAddbw = 3*1024**3
        self.sanity_patterns = sn.all([
            # check the job status:
            sn.assert_found('loop complete.', self.stdout),
            # check the tool's version (2019=591264, 2018=551025):
            sn.assert_eq(sn.extractsingle(
                r'I*.\(build\s(?P<toolsversion>\d+)\s*.',
                self.version_rpt, 'toolsversion'), toolsversion),
            # --- roofline boundaries:
            # check --report=roofs (L1 bandwidth):
            sn.assert_reference(sn.extractsingle(
                r'^L1\sbandwidth\s\(single-threaded\)\s+(?P<L1bw>\d+)\s+'
                r'memory$', self.roofline_ref, 'L1bw', int),
                L1bw, -0.08, 0.08),
            # check --report=roofs (L2 bandwidth):
            sn.assert_reference(sn.extractsingle(
                r'^L2\sbandwidth\s\(single-threaded\)\s+(?P<L2bw>\d+)\s+'
                r'memory$', self.roofline_ref, 'L2bw', int),
                L2bw, -0.08, 0.08),
            # check --report=roofs (L3 bandwidth):
            sn.assert_reference(sn.extractsingle(
                r'^L3\sbandwidth\s\(single-threaded\)\s+(?P<L3bw>\d+)\s+'
                r'memory$', self.roofline_ref, 'L3bw', int),
                L3bw, -0.08, 0.08),
            # check --report=roofs (DP FMA):
            sn.assert_reference(sn.extractsingle(
                r'^DP Vector FMA Peak\s\(single-threaded\)\s+'
                r'(?P<DPfmabw>\d+)\s+compute$', self.roofline_ref,
                'DPfmabw', int), DPfmabw, -0.08, 0.08),
            # check --report=roofs (DP Add):
            sn.assert_reference(sn.extractsingle(
                r'^DP Vector Add Peak\s\(single-threaded\)\s+'
                r'(?P<DPaddbw>\d+)\s+compute$', self.roofline_ref,
                'DPaddbw', int), DPaddbw, -0.08, 0.08),
            # check --report=roofs (Scalar Add):
            sn.assert_reference(sn.extractsingle(
                r'^Scalar Add Peak\s\(single-threaded\)\s+'
                r'(?P<ScalarAddbw>\d+)\s+compute$', self.roofline_ref,
                'ScalarAddbw', int), ScalarAddbw, -0.08, 0.08),
            # --- check Arithmetic_intensity:
            sn.assert_reference(sn.extractsingle(
                r'^returned\sAI\sgap\s=\s(?P<Intensity>.*)', self.roofline_rpt,
                'Intensity', float), 0.0, -0.01, 0.01),
            # --- check GFLOPS:
            sn.assert_reference(sn.extractsingle(
                r'^returned\sGFLOPS\sgap\s=\s(?P<Flops>.*)', self.roofline_rpt,
                'Flops', float), 0.0, -0.01, 0.01),
        ])
        self.maintainers = ['JG']
        self.tags = {'production'}

    def setup(self, partition, environ, **job_opts):
        super().setup(partition, environ, **job_opts)
        environ_name = self.current_environ.name
        prgenv_flags = self.prgenv_flags[environ_name]
        self.build_system.cxxflags = prgenv_flags
        launcher_cmd = ' '.join(self.job.launcher.command(self.job))
        self.post_run = [
            # collecting the performance data for the roofline model is a 2
            # steps process:
            '%s %s --collect tripcounts --flop --project-dir=%s '
            '--search-dir src:rp=. --data-limit=0 --no-auto-finalize '
            '--trace-mpi -- %s' %
            (launcher_cmd, self.executable, self.roofdir,
             self.target_executable),
            # check tool's version:
            'advixe-cl -V &> %s' % self.version_rpt,
            # "advixe-cl --report" looks for e000/ in the output directory;
            # if not found, it will fail with:
            # IOError: Survey result cannot be loaded
            'cd %s;ln -s nid* e000;cd -' % self.roofdir,
            # report reference values/boundaries (roofline_ref):
            'advixe-cl --report=roofs --project-dir=%s &> %s' %
            (self.roofdir, self.roofline_ref),
            'python2 API/cscs.py %s &> %s' % (self.roofdir, self.roofline_rpt),
            'touch the_end',
            # 'advixe-cl --format=csv' seems to be not working (empty report),
            # keeping as reference for a future check:
            #   'advixe-cl --show-all-columns -csv-delimiter=";"'
            #   ' --report=tripcounts --format=csv --project-dir=%s &> %s'
            # This can be used instead (see advisor/config/report/roofs.tmpl):
            #   'advixe-cl --report custom --report-template ./TEMPL/cscs.tmpl'
            #   ' --project-dir=%s &> %s'
        ]
