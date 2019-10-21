import os

import reframe as rfm
import reframe.utility.sanity as sn


class OpenfoamExtendBaseTest(rfm.RunOnlyRegressionTest):
    def __init__(self):
        dirname = self.name[0].lower() + self.name[1:]
        self.name = 'OpenfoamExtend_' + self.name
        self.executable = dirname
        self.sourcesdir = os.path.join(self.current_system.resourcesdir,
                                       'OpenFOAM-Extend', dirname)

        # OpenFOAM-Extend currently runs only on Leone
        self.valid_systems = ['leone:normal']
        self.valid_prog_environs = ['PrgEnv-gnu']
        self.modules = ['OpenFOAM-Extend/4.0-foss-2016b']

        self.num_tasks = 1
        self.num_tasks_per_node = 1
        self.num_cpus_per_task  = 1

        self.sanity_patterns = sn.assert_found(r'^\s*[Ee]nd', self.stdout)

        if self.num_tasks > 1:
            self.sanity_patterns = sn.assert_found(
                r'Finalising parallel run', self.stdout)

        self.maintainers = ['MKr']
        self.tags = {'scs', 'production', 'external-resources'}
        self.pre_run = ['source $FOAM_INST_DIR/foam-extend-4.0/etc/bashrc']


@rfm.simple_test
class BlockMesh(OpenfoamExtendBaseTest):
    def __init__(self):
        super().__init__()
        self.descr = 'OpenFOAM-Extend blockMesh from the dambreak tutorial'


@rfm.simple_test
class SnappyHexMesh(OpenfoamExtendBaseTest):
    def __init__(self):
        super().__init__()
        self.descr = ('OpenFOAM-Extend check of snappyHexMesh: '
                      'motorbike tutorial')


@rfm.simple_test
class SimpleFoam(OpenfoamExtendBaseTest):
    def __init__(self):
        super().__init__()
        self.descr = 'OpenFOAM-Extend check of simpleFoam: motorbike tutorial'
        self.executable_opts = ['-parallel']
        self.num_tasks = 6
        self.num_tasks_per_node = 6
        result = sn.extractall(
            r'time step continuity errors : '
            r'\S+\s\S+ = \S+\sglobal = (?P<res>-?\S+),',
            self.stdout, 'res', float)
        self.sanity_patterns = sn.all(
            sn.map(lambda x: sn.assert_lt(abs(x), 5.e-04), result))


@rfm.simple_test
class SetFields(OpenfoamExtendBaseTest):
    def __init__(self):
        super().__init__()
        self.descr = 'OpenFOAM-Extend  check of setFields: dambreak tutorial'


@rfm.simple_test
class InterMixingFoam(OpenfoamExtendBaseTest):
    def __init__(self):
        super().__init__()
        self.descr = ('OpenFOAM-Extend  check of interMixingFoam: '
                      'dambreak tutorial')
        self.sanity_patterns = sn.assert_eq(sn.count(sn.findall(
            r'Air phase volume fraction', self.stdout)), 2944)


@rfm.simple_test
class BuoyantBoussinesqSimpleFoam(OpenfoamExtendBaseTest):
    def __init__(self):
        super().__init__()
        self.descr = ('OpenFOAM-Extend check buoyantBoussinesqSimpleFoam: '
                      'hotRoom test')
        self.executable = 'buoyantBoussinesqSimpleFoam'
        result = sn.extractall(r'\sglobal\s=\s(?P<res>\S+),',
                               self.stdout, 'res', float)
        self.sanity_patterns = sn.all(
            sn.map(lambda x: sn.assert_lt(abs(x), 1.e-17), result))


@rfm.simple_test
class LaplacianFoam(OpenfoamExtendBaseTest):
    def __init__(self):
        super().__init__()
        self.descr = 'OpenFOAM-Extend check of setFields: flange tutorial'


@rfm.simple_test
class FoamToEnsight(OpenfoamExtendBaseTest):
    def __init__(self):
        super().__init__()
        self.descr = 'OpenFOAM-Extend check of setFields: flange tutorial'


@rfm.simple_test
class SetSet(OpenfoamExtendBaseTest):
    def __init__(self):
        super().__init__()
        self.descr = ('OpenFOAM-Extend check of setFields: '
                      'multi region heater tutorial')
        self.executable_opts  = ['-batch makeCellSets.setSet']


@rfm.simple_test
class SetsToZones(OpenfoamExtendBaseTest):
    def __init__(self):
        super().__init__()
        self.descr = ('OpenFOAM-Extend check of setFields: '
                      'multi region heater tutorial')
        self.executable_opts  = ['-noFlipMap']


@rfm.simple_test
class SplitMeshRegions(OpenfoamExtendBaseTest):
    def __init__(self):
        super().__init__()
        self.descr = ('OpenFOAM-Extend check of setFields: '
                      'multi region heater tutorial')
        self.executable_opts  = ['-cellZones', '-overwrite']


@rfm.simple_test
class DecomposePar(OpenfoamExtendBaseTest):
    def __init__(self):
        super().__init__()
        self.descr = ('OpenFOAM-Extend check of reconstructPar: '
                      'multiRegionHeater test')
        self.executable_opts  = ['-region heater']


@rfm.simple_test
class ChtMultiRegionSimpleFoam(OpenfoamExtendBaseTest):
    def __init__(self):
        super().__init__()
        self.descr = ('OpenFOAM-Extend check of reconstructPar: '
                      'multiRegionHeater test')
        self.executable_opts = ['-parallel']
        self.num_tasks          = 4
        self.num_tasks_per_node = 4

        result = sn.extractall(r'\sglobal\s=\s(?P<res>-?\S+),',
                               self.stdout, 'res', float)[-5:]
        self.sanity_patterns = sn.all(
            sn.map(lambda x: sn.assert_lt(abs(x), 1.e-04), result))


@rfm.simple_test
class ReconstructPar(OpenfoamExtendBaseTest):
    def __init__(self):
        super().__init__()
        self.descr = ('OpenFOAM-Extend check of reconstructPar: '
                      'multiRegionHeater test')
        self.executable_opts = ['-latestTime', '-region heater']
        self.readonly_files  = ['processor0',
                                'processor1',
                                'processor2',
                                'processor3']
        self.sanity_patterns = sn.assert_found(r'Time = 1000', self.stdout)
