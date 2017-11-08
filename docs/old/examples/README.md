# Example Regression Tests

This is a minimal set of regression tests that highlight some features of ReFrame.
The list is not meant to be extensive. But rather show the difficulty of generating collections of similar tests.


# Hello World

This is a simple and minimal *Hello World!* program used for the unit testing.
```python
# File: hello_world.py
# Generates several regression tests. The actual number depends on the number of programming environments defined in the system
import re

from reframe.core.pipeline import RegressionTest
from reframe.core.environments import *

class HelloTest(RegressionTest):
    def __init__(self, **kwargs):
        super().__init__('hellocheck', os.path.dirname(__file__), **kwargs)
        self.descr = 'C Hello World test'

        # All available systems are supported
        self.valid_systems = [ '*' ]

        # All programming environments  are supported
        self.valid_prog_environs = [ '*' ]
        self.sourcepath = 'hello.c'

        # Generic Tags
        self.tags = { 'foo', 'bar' }

        # Sanity pattern to match
        self.sanity_patterns = {
            '-' : { 'Hello, World\!' : [] }
        }
        self.maintainers = [ 'YY' ]

def _get_checks(**kwargs):
    return [ HelloTest(**kwargs) ]
```
# CUDA Regression Tests

This is a collection of example CUDA Regression Tests written inside a single file that share the same base class `CudaCheck`.

```python
# File: cuda.py
# Generates 10 regression tests per system.
import os

from reframe.core.pipeline import RegressionTest

class CudaCheck(RegressionTest):
    def __init__(self, name, **kwargs):
        super().__init__('cuda_%s_check' % name,
                         os.path.dirname(__file__), **kwargs)
        self.valid_systems = [ 'daint:gpu', 'dom:gpu', 'kesch:cn' ]
        self.valid_prog_environs = [ 'PrgEnv-cray', 'PrgEnv-gnu' ]
        self.modules = [ 'cudatoolkit' ]
        self.maintainers = [ 'XX' ]
        self.num_gpus_per_node = 1
        self.tags = { 'production' }

    def compile(self):
        # Set nvcc flags
        self.current_environ.cxxflags = '-ccbin g++ -m64 -lcublas'
        super().compile()


class MatrixmulCublasCheck(CudaCheck):
    def __init__(self, **kwargs):
        super().__init__('matrixmulcublas', **kwargs)
        self.descr = 'Implements matrix multiplication using CUBLAS'
        self.sourcepath = 'matrixmulcublas.cu'
        self.sanity_patterns = {
            '-' : {
                'Comparing CUBLAS Matrix Multiply with CPU results: PASS' : []
            }
        }


class BandwidthCheck(CudaCheck):
    def __init__(self, **kwargs):
        super().__init__('bandwidth', **kwargs)
        self.descr = 'CUDA bandwidthTest compile and run'
        self.sourcepath = 'bandwidthTest.cu'
        self.sanity_patterns = {
            '-' : { 'Result = PASS' : [] }
        }


class DeviceQueryCheck(CudaCheck):
    def __init__(self, **kwargs):
        super().__init__('devicequery', **kwargs)
        self.descr = 'Queries the properties of the CUDA devices'
        self.sourcepath = 'devicequery.cu'
        self.sanity_patterns = {
            '-' : { 'Result = PASS' : [] }
        }


class ConcurrentKernelsCheck(CudaCheck):
    def __init__(self, **kwargs):
        super().__init__('concurrentkernels', **kwargs)
        self.descr = 'Use of streams for concurrent execution'
        self.sourcepath = 'concurrentkernels.cu'
        self.sanity_patterns = {
            '-' : { 'Test passed' : [] }
        }


class SimpleMPICheck(CudaCheck):
    def __init__(self, **kwargs):
        super().__init__('simplempi', **kwargs)
        self.descr = 'Simple example demonstrating how to use MPI with CUDA'
        self.sourcepath = 'simplempi/'
        self.executable = 'simplempi'
        self.num_tasks = 2
        self.num_tasks_per_node = 2
        self.sanity_patterns = {
            '-' : { 'Result = PASS' : [] }
        }
        if self.current_system.name == 'kesch':
            self.variables = { 'G2G' : '0' }
            self.num_gpus_per_node = 2
        else:
            self.variables = { 'CRAY_CUDA_MPS' : '1' }


def _get_checks(**kwargs):
    return [ BandwidthCheck(**kwargs),
             ConcurrentKernelsCheck(**kwargs),
             DeviceQueryCheck(**kwargs),
             MatrixmulCublasCheck(**kwargs),
             SimpleMPICheck(**kwargs) ]
```

# OpenACC Regression Tests
This example shows how to differentiate behavior based on the current system and programming environment.

```python
# File: openacc.py
# Generates 2 regression tests per system.
import os

from reframe.core.pipeline import RegressionTest
from reframe.utility.functions import standard_threshold

class OpenACCFortranCheck(RegressionTest):
    def __init__(self, **kwargs):
        super().__init__('testing_fortran_openacc',
                         os.path.dirname(__file__), **kwargs)

        self.valid_systems = [ 'daint:gpu', 'dom:gpu' ]
        self.valid_prog_environs = [ 'PrgEnv-cray', 'PrgEnv-pgi' ]

        self.sourcepath = 'vecAdd_openacc.f90'
        self.num_gpus_per_node = 1

        self.sanity_patterns = {
            '-' : {
                'final result:\s+(?P<result>\d+\.?\d*)' : [
                    ('result', float,
                     lambda value, **kwargs: \
                         standard_threshold(value, (1., -1e-5, 1e-5)))
                ],
            }
        }
        self.modules = [ 'craype-accel-nvidia60' ]
        self.maintainers = [ 'XX' ]
        self.tags = { 'production' }

        # Change the compiler version for dom because the default installed version is broken
        if self.current_system.name == 'dom':
            self.modules += [ 'pgi/16.7.0' ]

    def setup(self, system, environ, **job_opts):
        # changing the compiler flags based on the compiler
        if environ.name == 'PrgEnv-cray':
            environ.fflags = '-hacc -hnoomp'
        elif environ.name == 'PrgEnv-pgi':
            environ.fflags = '-acc -ta=tesla:cuda8.0'

        super().setup(system, environ, **job_opts)


def _get_checks(**kwargs):
    return [ OpenACCFortranCheck(**kwargs) ]
```

# Application Regression Tests

This is an example application test for [CP2K](https://www.cp2k.org/). It uses the `RunOnlyRegressionTest` class and takes advantage of the [sanity and performance patterns](/writing_checks/#output-parsing-and-performance-assessment) using [statefull parsers](/writing_checks/#stateful-parsing-of-output).

```python
# File: cp2k.py
# Generates 2 regression tests per system.
import os
import reframe.settings as settings

from reframe.core.pipeline import RunOnlyRegressionTest
from reframe.utility.functions import standard_threshold
from reframe.utility.parsers import StatefulParser, CounterParser

class Cp2kCheck(RunOnlyRegressionTest):
    def __init__(self, check_name, check_descr, **kwargs):
        super().__init__(check_name, os.path.dirname(__file__), **kwargs)
        self.descr = check_descr
        self.valid_prog_environs = [ 'PrgEnv-gnu' ]

        self.executable = 'cp2k.psmp'
        self.executable_opts = [ 'H2O-256.inp' ]

        self.sanity_parser = CounterParser(10, exact=True)
        self.sanity_parser.on()
        self.sanity_patterns = {
            '-' : {
                '(?P<t_count_steps>STEP NUM)' : [
                    ('t_count_steps', str, self.sanity_parser.match)
                ],
                '(?P<c_count_steps>PROGRAM STOPPED IN)' : [
                    ('c_count_steps', str, self.sanity_parser.match_eof)
                ]
            }
        }

        self.perf_parser = StatefulParser(standard_threshold)
        self.perf_patterns = {
            '-' : {
                '(?P<perf_section>T I M I N G)' : [
                    ('perf_section', str, self.perf_parser.on)
                ],
                '^ CP2K(\s+[\d\.]+){4}\s+(?P<perf>\S+)' : [
                    ('perf', float, self.perf_parser.match)
                ]
            }
        }

        if self.current_system.name == 'dom':
            self.num_tasks = 48
        else:
            self.num_tasks = 128

        self.num_tasks_per_node = 8
        self.maintainers = [ 'XX' ]

        self.tags = { 'production', 'maintenance', 'scs' }
        self.strict_check = False
        self.reference = {
            'dom:gpu' : {
                'perf' : (258, None, 0.15)
            },
            'dom:mc' : {
                'perf' : (340, None, 0.15)
            },
            'daint:gpu' : {
                'perf' : (130, None, 0.15)
            },
            'daint:mc' : {
                'perf' : (135, None, 0.15)
            },
            '*' : {
                'perf_section' : None,
            }
        }

        self.modules = [ 'CP2K' ]
        self.num_gpus_per_node = 0


class Cp2kCheckCpu(Cp2kCheck):
    def __init__(self, **kwargs):
        super().__init__('cp2k_cpu_check', 'CP2K check CPU', **kwargs)
        self.valid_systems = [ 'daint:mc', 'dom:mc' ]
        self.num_gpus_per_node = 0


class Cp2kCheckGpu(Cp2kCheck):
    def __init__(self, **kwargs):
        super().__init__('cp2k_gpu_check', 'CP2K check GPU', **kwargs)
        self.valid_systems = [ 'daint:gpu', 'dom:gpu' ]
        self.variables = { 'CRAY_CUDA_MPS' : '1' }
        self.modules = [ 'CP2K' ]
        self.num_gpus_per_node = 1


def _get_checks(**kwargs):
    return [ Cp2kCheckCpu(**kwargs),
             Cp2kCheckGpu(**kwargs) ]
```

# Compile-only Regression Tests

This is a collection of compile-only regression tests. This example also shows the use of a custom base class (`LibSciResolveBaseTest`) that is reused to group sets of tests.

```python
# File: lib_resolve.py
# Generates 7 regression tests per system.
import re
import os

from reframe.core.pipeline import CompileOnlyRegressionTest

class LibSciResolveBaseTest(CompileOnlyRegressionTest):
    def __init__(self, name, **kwargs):
        super().__init__(name, os.path.dirname(__file__), **kwargs)
        self.sourcepath = 'libsci_resolve.f90'
        self.valid_systems = [ 'daint:login', 'dom:login' ]

      self.flags = ''

    def compile(self):
        self.current_environ.fflags += self.flags
        super().compile()

    def _find_symbol_definition(self, symbol):
        file = open(self.stderr, 'rt')

        # returning a list because for some tests there are
        # multiple definitions of pdgemm_
        ret = None
        for line in file:
            found = re.search('(?P<lib>\S+): '
                              'definition of %s' % symbol, line)
            if found:
                ret = os.path.basename(found.group('lib'))
                break

        file.close()
        return ret


class Nvidia35ResolveTest(LibSciResolveBaseTest):
    def __init__(self, module_name, **kwargs):
        super().__init__('accel_nvidia35_resolves_to_libsci_acc_%s' %
                         module_name.replace('-', '_'), **kwargs)

        self.descr = 'Module %s resolves libsci_acc' % module_name
        self.flags = ' -Wl,-ypdgemm_ '

        self.module_name = module_name
        self.module_version = {
            'craype-accel-nvidia20' : 'nv20',
            'craype-accel-nvidia35' : 'nv35',
            'craype-accel-nvidia60' : 'nv60'
        }
        self.compiler_version = {
            'dom'    : '49',
            'daint'  : '49',
        }
        self.compiler_version_default = '49'
        self.modules = [ module_name ]
        self.valid_prog_environs = [ 'PrgEnv-cray', 'PrgEnv-gnu' ]

        self.prgenv_names = {
            'PrgEnv-cray' : 'cray',
            'PrgEnv-gnu'  : 'gnu'
        }

        self.maintainers = [ 'XX' ]
        self.tags = { 'production' }


    def check_sanity(self):
        lib_name = self._find_symbol_definition('pdgemm')
        if not lib_name:
            return False

        # here lib_name is in the format: libsci_acc_gnu_48_nv35.so or
        #                                 libsci_acc_cray_nv35.so
        lib_name = re.search('libsci_acc_(?P<prgenv>[A-Za-z]+)_'
                            '((?P<cver>[A-Za-z0-9]+)_)?'
                            '(?P<version>\S+)'
                            '(?=(\.a)|(\.so))', lib_name)

        if not lib_name:
            return False

        prgenv = self.prgenv_names[self.current_environ.name]
        cver   = self.compiler_version.get(self.current_system.name,
                                           self.compiler_version_default)

        if lib_name.group('prgenv') != prgenv:
            return False
        elif self.current_environ.name == 'PrgEnv-cray' and \
             lib_name.group('cver'):
            return False
        elif self.current_environ.name == 'PrgEnv-gnu' and \
             lib_name.group('cver') != cver :
            return False
        elif lib_name.group('version') != self.module_version[self.module_name]:
            return False

        return True


class MKLResolveTest(LibSciResolveBaseTest):
    def __init__(self, **kwargs):
        super().__init__('mkl_resolves_to_mkl_intel', **kwargs)

        self.descr = '-mkl Resolves to MKL'
        self.flags = ' -Wl,-ydgemm_ -mkl'

        self.valid_prog_environs = [ 'PrgEnv-intel' ]
        self.maintainers = [ 'XX' ]
        self.tags = { 'production' }


    def check_sanity(self):
        lib_name = self._find_symbol_definition('dgemm')
        # here lib_name is in the format: libmkl_intel_lp64.a(_dgemm_lp64.o)
        found = re.search('libmkl_(?P<prgenv>[A-Za-z]+)_'
                          '(?P<version>\S+)'
                          '(?=(.a)|(.so))', lib_name)
        # interesting enough, on Dora the linking here is static.
        # So there is REAL need for the end term (?=(.a)|(.so)).

        if not found:
            return False

        # not sure if we need to check against the version here
        if found.group('prgenv') != 'intel':
            return False
        elif found.group('version') != 'lp64':
            return False

        return True


def _get_checks(**kwargs):
    """
    Returns an instance of the regression test to the calling framework. This is
    the entry point of the framework to the user-defined checks. We could
    automatically detect regression checks in the future, so this function would
    not be necessary.
    """
    ret = [ MKLResolveTest(**kwargs) ]
    for module_name in [ 'craype-accel-nvidia20',
                         'craype-accel-nvidia35',
                         'craype-accel-nvidia60' ]:
        ret.append(Nvidia35ResolveTest(module_name, **kwargs))

    return ret
```