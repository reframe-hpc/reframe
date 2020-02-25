# Copyright 2016-2020 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import os

import reframe as rfm
import reframe.utility.sanity as sn


@rfm.parameterized_test(*[[repeat, toolversion, datalayout]
                          for repeat in ['100000']
                          for toolversion in ['597843']
                          for datalayout in ['G3_AOS_SCALAR', 'G3_SOA_SCALAR',
                                             'G3_AOS_VECTOR', 'G3_SOA_VECTOR']
                          ])
class IntelRooflineAdvisorTest(rfm.RegressionTest):
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
        G3_AOS_SCALAR: gflops,  2.79 arithmetic_intensity', 0.166 380ms <- slow
        G3_AOS_VECTOR: gflops,  3.79 arithmetic_intensity', 0.125 143ms
        G3_SOA_SCALAR: gflops,  2.79 arithmetic_intensity', 0.166 351ms
        G3_SOA_VECTOR: gflops, 10.62 arithmetic_intensity', 0.166  57ms <- fast
    '''

    def __init__(self, repeat, toolversion, datalayout):
        self.descr = 'Roofline Analysis test with Intel Advisor'
        # for reference: advisor/2019 was failing on dom with:
        # "Exceeded job memory limit" (webrt#36087)
        self.valid_systems = ['daint:mc', 'dom:mc']
        # Reporting MFLOPS is not available on Intel Haswell cpus, see
        # https://www.intel.fr/content/dam/www/public/us/en/documents/manuals/
        # 64-ia-32-architectures-software-developer-vol-1-manual.pdf
        self.valid_prog_environs = ['PrgEnv-intel']
        self.modules = ['advisor/2019_update4']
        # Testing with advisor/2018 (build 551025) fails with:
        #    roof.dir/nid00753.000/trc000/trc000.advixe
        #    Application exit code: 139
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
        self.roofline_rpt = '%s.rpt' % self.target_executable
        self.version_rpt = 'Intel_Advisor_version.rpt'
        self.roofline_ref = 'Intel_Advisor_roofline_reference.rpt'
        self.prebuild_cmd = [
            'patch -s < ADVISOR/roofline_template.patch',
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
        # - Reference roofline boundaries for Intel Broadwell CPU (E5-2695 v4):
        L1bw = 293  # *1024**3
        L2bw = 79   # *1024**3
        L3bw = 33   # *1024**3
        DPfmabw = 45*1024**3
        DPaddbw = 12*1024**3
        ScalarAddbw = 3*1024**3
        # --- roofline (memory) boundaries from the tool:
        # DRAM Bandwidth (single node)             	   63206331080	 memory
        # DRAM Bandwidth                           	  125993278750	 memory
        # DRAM Bandwidth (single-threaded)         	   12715570803	 memory
        # L1 Bandwidth                             	11360856466728	 memory
        # Scalar L1 Bandwidth                      	 2648216636280	 memory
        # L1 bandwidth (single-threaded)           	  315579346298	 memory
        #                                                 ************
        # Scalar L1 bandwidth (single-threaded)    	   73561573230	 memory
        # L2 Bandwidth                             	 3102773429268	 memory
        # Scalar L2 Bandwidth                      	  921316779936	 memory
        # L2 bandwidth (single-threaded)           	   86188150813	 memory
        #                                                  ***********
        # Scalar L2 bandwidth (single-threaded)    	   25592132776	 memory
        # L3 Bandwidth                             	 1269637300440	 memory
        # Scalar L3 Bandwidth                      	  845928498744	 memory
        # L3 bandwidth (single-threaded)           	   35267702790	 memory
        #                                                  ***********
        # Scalar L3 bandwidth (single-threaded)    	   23498013854	 memory
        regex_roof_L1 = (r'^L1\sbandwidth\s\(single-threaded\)\s+(?P<L1bw>\d+)'
                         r'\s+memory$')
        regex_roof_L2 = (r'^L2\sbandwidth\s\(single-threaded\)\s+(?P<L2bw>\d+)'
                         r'\s+memory$')
        regex_roof_L3 = (r'^L3\sbandwidth\s\(single-threaded\)\s+(?P<L3bw>\d+)'
                         r'\s+memory$')
        roof_L1 = sn.round(sn.extractsingle(regex_roof_L1, self.roofline_ref,
                                            'L1bw', int) / 1024**3, 2)
        roof_L2 = sn.round(sn.extractsingle(regex_roof_L2, self.roofline_ref,
                                            'L2bw', int) / 1024**3, 3)
        roof_L3 = sn.round(sn.extractsingle(regex_roof_L3, self.roofline_ref,
                                            'L3bw', int) / 1024**3, 3)

        # --- roofline (compute) boundaries from the tool:
        # SP Vector FMA Peak                       	 2759741518342	compute
        # SP Vector FMA Peak (single-threaded)     	   98956234406	compute
        # DP Vector FMA Peak                       	 1379752337990	compute
        # DP Vector FMA Peak (single-threaded)     	   49563336304	compute
        #                                                  ***********
        # Scalar Add Peak                          	   93438527464	compute
        # Scalar Add Peak (single-threaded)        	    3289577753	compute
        #                                                   **********
        # SP Vector Add Peak                       	  689944922272	compute
        # SP Vector Add Peak (single-threaded)     	   24691445241	compute
        # DP Vector Add Peak                       	  344978547363	compute
        # DP Vector Add Peak (single-threaded)     	   12385333008	compute
        #                                                  ***********
        # Integer Scalar Add Peak                  	  228677310757	compute
        # Integer Scalar Add Peak (single-threaded)	    8055287031	compute
        # Int64 Vector Add Peak                    	  747457604632	compute
        # Int64 Vector Add Peak (single-threaded)  	   26300241032	compute
        # Int32 Vector Add Peak                    	 1494880413924	compute
        # Int32 Vector Add Peak (single-threaded)  	   52738180380	compute
        regex_roof_dpfma = (r'^DP Vector FMA Peak\s\(single-threaded\)\s+'
                            r'(?P<DPfmabw>\d+)\s+compute$')
        regex_roof_dpadd = (r'^DP Vector Add Peak\s\(single-threaded\)\s+'
                            r'(?P<DPaddbw>\d+)\s+compute$')
        regex_roof_scalaradd = (r'^Scalar Add Peak\s\(single-threaded\)\s+'
                                r'(?P<ScalarAddbw>\d+)\s+compute$')
        roof_dpfma = sn.extractsingle(regex_roof_dpfma, self.roofline_ref,
                                      'DPfmabw', int)
        roof_dpadd = sn.extractsingle(regex_roof_dpadd, self.roofline_ref,
                                      'DPaddbw', int)
        roof_scalaradd = sn.extractsingle(regex_roof_scalaradd,
                                          self.roofline_ref, 'ScalarAddbw',
                                          int)

        # - API output:
        # ('self_elapsed_time', 0.1)
        # ('self_memory_gb', 4.2496)
        # ('self_gb_s', 42.496)
        # ('self_gflop', 0.5312)
        # ('self_gflops', 5.312)
        # ('self_arithmetic_intensity', 0.125)
        # ('_self_gb_s', 42.495999999999995, 42.496)
        # ('_self_gflops', 5.311999999999999, 5.312)
        # ('_self_arithmetic_intensity', 0.125, 0.125)
        # ('gap _self_gb_s', -7.105427357601002e-15)
        # ('gap _self_gflops', -8.881784197001252e-16)
        # ('gap _self_arithmetic_intensity', 0.0)
        # returned AI gap = 0.0000000000000000
        # returned GFLOPS gap = -0.0000000000000009
        regex_ai_gap = r'^returned\sAI\sgap\s=\s(?P<Intensity>.*)'
        regex_ai_gflops = r'^returned\sGFLOPS\sgap\s=\s(?P<Flops>.*)'
        ai_gap = sn.extractsingle(regex_ai_gap, self.roofline_rpt, 'Intensity',
                                  float)
        ai_gflops = sn.extractsingle(regex_ai_gflops, self.roofline_rpt,
                                     'Flops', float)

        regex_toolversion = r'I*.\(build\s(?P<version>\d+)\s*.'
        found_toolversion = sn.extractsingle(regex_toolversion,
                                             self.version_rpt, 'version')
        self.sanity_patterns = sn.all([
            # check the job status:
            sn.assert_found('loop complete.', self.stdout),
            # check the tool's version (2019=591264, 2018=551025):
            sn.assert_eq(found_toolversion, toolversion),
            # --- roofline boundaries:
            # check --report=roofs (L1, L2 and L3 bandwidth):
            # sn.assert_reference(roof_L1, L1bw, -0.12, 0.08),
            # sn.assert_reference(roof_L2, L2bw, -0.12, 0.08),
            # sn.assert_reference(roof_L3, L3bw, -0.12, 0.08),
            # check --report=roofs (DP FMA, DP Add and Scalar Add):
            sn.assert_reference(roof_dpfma, DPfmabw, -0.12, 0.08),
            sn.assert_reference(roof_dpadd, DPaddbw, -0.12, 0.08),
            sn.assert_reference(roof_scalaradd, ScalarAddbw, -0.12, 0.08),
            # --- check Arithmetic_intensity:
            sn.assert_reference(ai_gap, 0.0, -0.01, 0.01),
            # --- check GFLOPS:
            sn.assert_reference(ai_gflops, 0.0, -0.01, 0.01),
        ])

        # --performance-report:
        regex_mseconds = r'elapsed time: (?P<msec>\d+)ms'
        regex_ai = r'^\(\'self_arithmetic_intensity\', (?P<AI>\d+.\d+)\)'
        regex_gbs = r'^\(\'self_gb_s\', (?P<gbs>\d+.\d+)\)'
        regex_gflops = r'^\(\'self_gflops\', (?P<gflops>\d+.\d+)\)'
        mseconds = sn.extractsingle(regex_mseconds, self.stdout,
                                    'msec', int)
        arithmetic_intensity = sn.extractsingle(regex_ai, self.roofline_rpt,
                                                'AI', float)
        bandwidth = sn.extractsingle(regex_gbs, self.roofline_rpt,
                                     'gbs', float)
        gflops = sn.extractsingle(regex_gflops, self.roofline_rpt,
                                  'gflops', float)
        self.perf_patterns = {
            'Elapsed': mseconds,
            'ArithmeticIntensity': arithmetic_intensity,
            'GFlops': gflops,
            'Bandwidth': bandwidth,
            'roof_L1': roof_L1,
            'roof_L2': roof_L2,
            'roof_L3': roof_L3,
        }
        self.reference = {
            '*': {
                'Elapsed': (0, None, None, 'ms'),
                'ArithmeticIntensity': (0, None, None, ''),
                'GFlops': (0, None, None, 'GFLOPs/s'),
                'Bandwidth': (0, None, None, 'GB/s'),
                'roof_L1': (L1bw, -0.12, 0.08, 'GB/s'),
                'roof_L2': (L2bw, -0.12, 0.08, 'GB/s'),
                'roof_L3': (L3bw, -0.12, 0.08, 'GB/s'),
            }
        }

        self.maintainers = ['JG', 'MKr']
        self.tags = {'production', 'external-resources'}

    def setup(self, partition, environ, **job_opts):
        super().setup(partition, environ, **job_opts)
        environ_name = self.current_environ.name
        prgenv_flags = self.prgenv_flags[environ_name]
        self.build_system.cxxflags = prgenv_flags
        launcher_cmd = ' '.join(self.job.launcher.command(self.job))
        self.post_run = [
            # --- collecting the performance data for the roofline model is a 2
            # steps process:
            '%s %s --collect tripcounts --flop --project-dir=%s '
            '--search-dir src:rp=. --data-limit=0 --no-auto-finalize '
            '--trace-mpi -- %s' %
            (launcher_cmd, self.executable, self.roofdir,
             self.target_executable),
            # --- check tool's version:
            'advixe-cl -V &> %s' % self.version_rpt,
            # "advixe-cl --report" looks for e000/ in the output directory;
            # if not found, it will fail with:
            # IOError: Survey result cannot be loaded
            'cd %s;ln -s nid* e000;cd -' % self.roofdir,
            # --- report reference values/boundaries (roofline_ref):
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
