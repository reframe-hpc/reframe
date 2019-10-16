import os

import reframe as rfm
import reframe.utility.sanity as sn


class OpenFOAMBaseTest(rfm.RunOnlyRegressionTest):
    def __init__(self):
        dirname = self.name[0].lower() + self.name[1:]
        self.name = 'OpenFoam_' + self.name
        self.executable = dirname
        self.sourcesdir = os.path.join(self.current_system.resourcesdir,
                                       'OpenFOAM', dirname)

        # OpenFOAM currently runs only on Leone
        self.valid_systems = ['leone:normal']
        self.valid_prog_environs = ['PrgEnv-gnu']
        self.modules = ['OpenFOAM/4.1-foss-2016b']

        self.num_tasks = 1
        self.num_tasks_per_node = 1
        self.num_cpus_per_task  = 1

        self.maintainers = ['MKr']
        self.tags = {'scs', 'production', 'external-resources'}

        self.pre_run = ['source $FOAM_BASH']


@rfm.simple_test
class BlockMesh(OpenFOAMBaseTest):
    def __init__(self):
        super().__init__()
        self.descr = 'OpenFOAM blockMesh from the dambreak tutorial'
        self.sanity_patterns = sn.assert_found(r'^\s*[Ee]nd', self.stdout)


@rfm.simple_test
class BuoyantBoussinesqSimpleFoam(OpenFOAMBaseTest):
    def __init__(self):
        super().__init__()
        self.descr = ('OpenFOAM check of buoyantBoussinesqSimpleFoam: '
                      'hotroom tutorial')
        self.executable = 'buoyantBoussinesqSimpleFoam'
        residual = sn.extractall(r'\sglobal\s=\s(?P<res>\S+),', self.stdout,
                                 'res', float)
        self.sanity_patterns = sn.all(sn.chain(
            sn.map(lambda x: sn.assert_lt(x, 1.e-17), residual),
            [sn.assert_found(r'^\s*[Ee]nd', self.stdout)]))


@rfm.simple_test
class CheckMesh(OpenFOAMBaseTest):
    def __init__(self):
        super().__init__()
        self.descr = 'OpenFOAM check of checkMesh: flange tutorial'
        self.executable_opts = ['-latestTime', '-allTopology',
                                '-allGeometry', '-parallel']
        self.num_tasks = 8
        self.num_tasks_per_node = 8
        self.sanity_patterns = sn.all([
            sn.assert_found('Finalising parallel run', self.stdout),
            sn.assert_found(r'^\s*[Ee]nd', self.stdout)
        ])


@rfm.simple_test
class ChtMultiRegionSimpleFoam(OpenFOAMBaseTest):
    def __init__(self):
        super().__init__()
        self.descr = ('OpenFOAM check of chtMultiRegionSimpleFoam:'
                      ' heatexchanger tutorial')
        self.executable_opts = ['-parallel']
        self.num_tasks = 4
        self.num_tasks_per_node = 4
        residual = sn.extractall(r'\sglobal\s=\s(?P<res>\S+),', self.stdout,
                                 'res', float)[-10:]
        self.sanity_patterns = sn.all(sn.chain(
            sn.map(lambda x: sn.assert_lt(x, 1.e-03), residual),
            [sn.assert_eq(10, sn.count(residual)),
             sn.assert_found('Finalising parallel run', self.stdout),
             sn.assert_found(r'^\s*[Ee]nd', self.stdout)]))


@rfm.simple_test
class CollapseEdges(OpenFOAMBaseTest):
    def __init__(self):
        super().__init__()
        self.descr = 'OpenFOAM check of collapseEdges: flange tutorial'
        self.executable_opts = ['-latestTime', '-collapseFaces', '-parallel']
        self.num_tasks = 8
        self.num_tasks_per_node = 8
        self.sanity_patterns = sn.all(
            [sn.assert_found('Finalising parallel run', self.stdout),
             sn.assert_found(r'^\s*[Ee]nd', self.stdout)])


@rfm.simple_test
class CreateBaffles(OpenFOAMBaseTest):
    def __init__(self):
        super().__init__()
        self.descr = 'OpenFOAM check of createBaffles: heatexchanger tutorial'
        self.executable_opts = ['-region air', '-overwrite']
        self.sanity_patterns = sn.assert_found(r'^\s*[Ee]nd', self.stdout)


@rfm.simple_test
class DecomposePar(OpenFOAMBaseTest):
    def __init__(self):
        super().__init__()
        self.descr = 'OpenFOAM check of decomposePar: heatexchanger tutorial'
        self.executable_opts = ['-region air']
        self.sanity_patterns = sn.assert_found(r'^\s*[Ee]nd', self.stdout)


@rfm.simple_test
class FoamyHexMesh(OpenFOAMBaseTest):
    def __init__(self):
        super().__init__()
        self.descr = 'OpenFOAM check of foamyHexMesh: motorbike tutorial'
        self.executable_opts = ['-parallel']
        self.num_tasks = 8
        self.num_tasks_per_node = 8
        self.sanity_patterns = sn.all(
            [sn.assert_found('Time = 100\n', self.stdout),
             sn.assert_found('Finalising parallel run', self.stdout),
             sn.assert_found(r'^\s*[Ee]nd', self.stdout)])


@rfm.simple_test
class InterMixingFoam(OpenFOAMBaseTest):
    def __init__(self):
        super().__init__()
        self.descr = 'OpenFOAM check of interMixingFoam: dambreak tutorial'
        self.sanity_patterns = sn.all([
            sn.assert_eq(
                sn.count(sn.findall('(?P<line>Air phase volume fraction)',
                                    self.stdout)), 2534),
            sn.assert_found(r'^\s*[Ee]nd', self.stdout)
        ])


@rfm.simple_test
class PatchSummary(OpenFOAMBaseTest):
    def __init__(self):
        super().__init__()
        self.descr = 'OpenFOAM check of patchSummary: motorbike tutorial'
        self.executable_opts = ['-parallel']
        self.num_tasks = 6
        self.num_tasks_per_node = 6
        self.sanity_patterns = sn.all([
            sn.assert_found('Finalising parallel run', self.stdout),
            sn.assert_found(r'^\s*[Ee]nd', self.stdout)
        ])


@rfm.simple_test
class PimpleFoam(OpenFOAMBaseTest):
    def __init__(self):
        super().__init__()
        self.descr = 'OpenFOAM check of pimpleFoam: tjunction tutorial'
        residual = sn.extractall(r'Solving for epsilon, \w+\s\w+\s=\s\d.\d+.\s'
                                 r'Final residual\s=\s(?P<res>-?\S+),',
                                 self.stdout, 'res', float)
        self.sanity_patterns = sn.all(sn.chain(
            sn.map(lambda x: sn.assert_lt(x, 5.e-05), residual),
            [sn.assert_found(r'^\s*[Ee]nd', self.stdout)],
        ))


@rfm.simple_test
class PotentialFoam(OpenFOAMBaseTest):
    def __init__(self):
        super().__init__()
        self.descr = 'OpenFOAM check of potentialFoam: motorbike tutorial'
        self.executable_opts = ['-parallel']
        self.num_tasks = 6
        self.num_tasks_per_node = 6
        residual = sn.extractall(r'Final residual = (?P<res>-?\S+),',
                                 self.stdout, 'res', float)[-5:]
        self.sanity_patterns = sn.all(
            sn.chain(sn.map(lambda x: sn.assert_lt(x, 1.e-07), residual),
                     [sn.assert_eq(5, sn.count(residual)),
                      sn.assert_found('Finalising parallel run', self.stdout),
                      sn.assert_found(r'^\s*[Ee]nd', self.stdout)])
        )


@rfm.simple_test
class ReconstructPar(OpenFOAMBaseTest):
    def __init__(self):
        super().__init__()
        self.descr = 'OpenFOAM check of reconstructPar: heatexchanger tutorial'
        self.executable_opts = ['-latestTime', '-region air']
        self.readonly_files  = ['processor0', 'processor1',
                                'processor2', 'processor3']
        self.sanity_patterns = sn.all([
            sn.assert_found('Time = 2000', self.stdout),
            sn.assert_found(r'^\s*[Ee]nd', self.stdout)
        ])


@rfm.simple_test
class ReconstructParMesh(OpenFOAMBaseTest):
    def __init__(self):
        super().__init__()
        self.descr = 'OpenFOAM check of reconstructParMesh: motorbike tutorial'
        self.executable_opts = ['-constant']
        self.readonly_files  = ['processor0', 'processor1', 'processor2',
                                'processor3', 'processor4', 'processor5']
        self.sanity_patterns = sn.assert_found(r'^\s*[Ee]nd', self.stdout)


@rfm.simple_test
class SetFields(OpenFOAMBaseTest):
    def __init__(self):
        super().__init__()
        self.descr = 'OpenFOAM check of setFields: dambreak tutorial'
        self.sanity_patterns = sn.assert_found(r'^\s*[Ee]nd', self.stdout)


@rfm.simple_test
class SimpleFoam(OpenFOAMBaseTest):
    def __init__(self):
        super().__init__()
        self.descr = 'OpenFOAM check of simpleFoam: motorbike tutorial'
        self.executable_opts = ['-parallel']
        self.num_tasks = 6
        self.num_tasks_per_node = 6
        self.sanity_patterns = sn.all([
            sn.assert_found('Finalising parallel run', self.stdout),
            sn.assert_found(r'^\s*[Ee]nd', self.stdout),
            sn.assert_lt(sn.abs(sn.extractsingle(
                r'time step continuity errors : \S+\s\S+ = \S+\s'
                r'global = (?P<res>-?\S+),',
                self.stdout, 'res', float)), 1.e-04)
        ])


@rfm.simple_test
class SnappyHexMesh(OpenFOAMBaseTest):
    def __init__(self):
        super().__init__()
        self.descr = 'OpenFOAM check of snappyHexMesh: motorbike tutorial'
        self.executable_opts = ['-overwrite', ' -parallel']
        self.num_tasks = 6
        self.num_tasks_per_node = 6
        self.sanity_patterns = sn.all([
            sn.assert_found('Finalising parallel run', self.stdout),
            sn.assert_found(r'^\s*[Ee]nd', self.stdout)
        ])


@rfm.simple_test
class SurfaceFeatureExtract(OpenFOAMBaseTest):
    def __init__(self):
        super().__init__()
        self.descr = ('OpenFOAM check of surfaceFeatureExtract: '
                      'motorbike tutorial')
        self.sanity_patterns = sn.assert_found(r'^\s*[Ee]nd', self.stdout)


@rfm.simple_test
class TopoSet(OpenFOAMBaseTest):
    def __init__(self):
        super().__init__()
        self.descr = 'OpenFOAM check of topoSet: heatexchanger tutorial'
        self.executable_opts = ['-region air', '-dict system/topoSetDict.1']
        self.sanity_patterns = sn.assert_found(r'^\s*[Ee]nd', self.stdout)
