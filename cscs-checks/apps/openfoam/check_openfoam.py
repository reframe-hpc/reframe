import os

import reframe.utility.sanity as sn
from reframe.core.pipeline import RunOnlyRegressionTest


class OpenFOAMBaseTest(RunOnlyRegressionTest):
    def __init__(self, check_name, check_descr, **kwargs):
        super().__init__('Openfoam_%s' % (check_name),
                         os.path.dirname(__file__), **kwargs)

        self.descr = check_descr
        self.executable = check_name
        self.sourcesdir = os.path.join(self.current_system.resourcesdir,
                                       'OpenFOAM', check_name)

        # OpenFOAM currently runs only on Leone
        self.valid_systems = ['leone:normal']
        self.valid_prog_environs = ['PrgEnv-gnu']
        self.modules = ['OpenFOAM/4.1-foss-2016b']

        self.num_tasks = 1
        self.num_tasks_per_node = 1
        self.num_cpus_per_task  = 1

        self.maintainers = ['MaKra']
        self.tags = {'scs', 'production'}

    def setup(self, partition, environ, **job_opts):
        super().setup(partition, environ, **job_opts)
        self.job.pre_run = ['source $FOAM_BASH']


class BlockMesh(OpenFOAMBaseTest):
    def __init__(self, **kwargs):
        super().__init__('blockMesh',
                         'OpenFOAM blockMesh from the dambreak tutorial',
                         **kwargs)

        self.sanity_patterns = sn.assert_found(r'^\s*[Ee]nd', self.stdout)


class BuoyantBoussinesqSimpleFoam(OpenFOAMBaseTest):
    def __init__(self, **kwargs):
        super().__init__(
            'buoyantBoussinesqSimpleFoam',
            'OpenFOAM check of buoyantBoussinesqSimpleFoam: hotroom tutorial',
            **kwargs)

        self.executable = 'buoyantBoussinesqSimpleFoam'
        residual = sn.extractall(r'\sglobal\s=\s(?P<res>\S+),', self.stdout,
                                 'res', float)
        self.sanity_patterns = sn.all(sn.chain(
            sn.map(lambda x: sn.assert_lt(x, 1.e-17), residual),
            [sn.assert_found(r'^\s*[Ee]nd', self.stdout)]))


class CheckMesh(OpenFOAMBaseTest):
    def __init__(self, **kwargs):
        super().__init__('checkMesh',
                         'OpenFOAM check of checkMesh: flange tutorial',
                         **kwargs)

        self.executable_opts = ['-latestTime', '-allTopology',
                                '-allGeometry', '-parallel']
        self.num_tasks = 8
        self.num_tasks_per_node = 8

        self.sanity_patterns = sn.all([
            sn.assert_found('Finalising parallel run', self.stdout),
            sn.assert_found(r'^\s*[Ee]nd', self.stdout)
        ])


class ChtMultiRegionSimpleFoam(OpenFOAMBaseTest):
    def __init__(self, **kwargs):
        super().__init__(
            'chtMultiRegionSimpleFoam',
            'OpenFOAM check of chtMultiRegionSimpleFoam:'
            ' heatexchanger tutorial',
            **kwargs)

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


class CollapseEdges(OpenFOAMBaseTest):
    def __init__(self, **kwargs):
        super().__init__('collapseEdges',
                         'OpenFOAM check of collapseEdges: flange tutorial',
                         **kwargs)

        self.executable_opts = ['-latestTime', '-collapseFaces', '-parallel']
        self.num_tasks = 8
        self.num_tasks_per_node = 8

        self.sanity_patterns = sn.all(
            [sn.assert_found('Finalising parallel run', self.stdout),
             sn.assert_found(r'^\s*[Ee]nd', self.stdout)])


class CreateBaffles(OpenFOAMBaseTest):
    def __init__(self, **kwargs):
        super().__init__(
            'createBaffles',
            'OpenFOAM check of createBaffles: heatexchanger tutorial',
            **kwargs)

        self.executable_opts = ['-region air', '-overwrite']

        self.sanity_patterns = sn.assert_found(r'^\s*[Ee]nd', self.stdout)


class DecomposePar(OpenFOAMBaseTest):
    def __init__(self, **kwargs):
        super().__init__(
            'decomposePar',
            'OpenFOAM check of decomposePar: heatexchanger tutorial',
            **kwargs)

        self.executable_opts = ['-region air']

        self.sanity_patterns = sn.assert_found(r'^\s*[Ee]nd', self.stdout)


class FoamyHexMesh(OpenFOAMBaseTest):
    def __init__(self, **kwargs):
        super().__init__('foamyHexMesh',
                         'OpenFOAM check of foamyHexMesh: motorbike tutorial',
                         **kwargs)

        self.executable_opts = ['-parallel']
        self.num_tasks = 8
        self.num_tasks_per_node = 8

        self.sanity_patterns = sn.all(
            [sn.assert_found('Time = 100\n', self.stdout),
             sn.assert_found('Finalising parallel run', self.stdout),
             sn.assert_found(r'^\s*[Ee]nd', self.stdout)])


class InterMixingFoam(OpenFOAMBaseTest):
    def __init__(self, **kwargs):
        super().__init__('interMixingFoam',
                         'OpenFOAM check of interMixingFoam:'
                         'dambreak tutorial',
                         **kwargs)

        self.sanity_patterns = sn.all(
            [sn.assert_eq(
                sn.count(sn.findall('(?P<line>Air phase volume fraction)',
                                    self.stdout)), 2534),
             sn.assert_found(r'^\s*[Ee]nd', self.stdout)])


class PatchSummary(OpenFOAMBaseTest):
    def __init__(self, **kwargs):
        super().__init__(
            'patchSummary',
            'OpenFOAM check of patchSummary: motorbike tutorial',
            **kwargs)

        self.executable_opts = ['-parallel']
        self.num_tasks = 6
        self.num_tasks_per_node = 6

        self.sanity_patterns = sn.all(
            [sn.assert_found('Finalising parallel run', self.stdout),
             sn.assert_found(r'^\s*[Ee]nd', self.stdout)])


class PimpleFoam(OpenFOAMBaseTest):
    def __init__(self, **kwargs):
        super().__init__('pimpleFoam',
                         'OpenFOAM check of pimpleFoam: tjunction tutorial',
                         **kwargs)

        residual = sn.extractall(r'Solving for epsilon, \w+\s\w+\s=\s\d.\d+.\s'
                                 r'Final residual\s=\s(?P<res>-?\S+),',
                                 self.stdout, 'res', float)
        self.sanity_patterns = sn.all(sn.chain(
            sn.map(lambda x: sn.assert_lt(x, 5.e-05), residual),
            [sn.assert_found(r'^\s*[Ee]nd', self.stdout)],
        ))


class PotentialFoam(OpenFOAMBaseTest):
    def __init__(self, **kwargs):
        super().__init__('potentialFoam',
                         'OpenFOAM check of potentialFoam: motorbike tutorial',
                         **kwargs)

        self.executable_opts = ['-parallel']
        self.num_tasks = 6
        self.num_tasks_per_node = 6
        residual = sn.extractall(r'Final residual = (?P<res>-?\S+),',
                                 self.stdout, 'res', float)[-5:]
        self.sanity_patterns = sn.all(
            sn.chain(sn.map(lambda x: sn.assert_lt(x, 1.e-07), residual),
                     [sn.assert_eq(5, sn.count(residual)),
                      sn.assert_found('Finalising parallel run', self.stdout),
                      sn.assert_found(r'^\s*[Ee]nd', self.stdout)]))


class ReconstructPar(OpenFOAMBaseTest):
    def __init__(self, **kwargs):
        super().__init__(
            'reconstructPar',
            'OpenFOAM check of reconstructPar: heatexchanger tutorial',
            **kwargs)

        self.executable_opts = ['-latestTime', '-region air']
        self.readonly_files  = ['processor0', 'processor1',
                                'processor2', 'processor3']
        self.sanity_patterns = sn.all(
            [sn.assert_found('Time = 2000', self.stdout),
             sn.assert_found(r'^\s*[Ee]nd', self.stdout)])


class ReconstructParMesh(OpenFOAMBaseTest):
    def __init__(self, **kwargs):
        super().__init__(
            'reconstructParMesh',
            'OpenFOAM check of reconstructParMesh: motorbike tutorial',
            **kwargs)

        self.executable_opts = ['-constant']
        self.readonly_files  = ['processor0', 'processor1', 'processor2',
                                'processor3', 'processor4', 'processor5']

        self.sanity_patterns = sn.assert_found(r'^\s*[Ee]nd', self.stdout)


class SetFields(OpenFOAMBaseTest):
    def __init__(self, **kwargs):
        super().__init__('setFields',
                         'OpenFOAM check of setFields: dambreak tutorial',
                         **kwargs)

        self.sanity_patterns = sn.assert_found(r'^\s*[Ee]nd', self.stdout)


class SimpleFoam(OpenFOAMBaseTest):
    def __init__(self, **kwargs):
        super().__init__('simpleFoam',
                         'OpenFOAM check of simpleFoam: motorbike tutorial',
                         **kwargs)

        self.executable_opts = ['-parallel']
        self.num_tasks = 6
        self.num_tasks_per_node = 6

        self.sanity_patterns = sn.all(
            [sn.assert_found('Finalising parallel run', self.stdout),
             sn.assert_found(r'^\s*[Ee]nd', self.stdout),
             sn.assert_lt(sn.abs(sn.extractsingle(
                 r'time step continuity errors : \S+\s\S+ = \S+\s'
                 r'global = (?P<res>-?\S+),',
                 self.stdout, 'res', float)), 1.e-04)])


class SnappyHexMesh(OpenFOAMBaseTest):
    def __init__(self, **kwargs):
        super().__init__('snappyHexMesh',
                         'OpenFOAM check of snappyHexMesh: motorbike tutorial',
                         **kwargs)

        self.executable_opts = ['-overwrite', ' -parallel']
        self.num_tasks = 6
        self.num_tasks_per_node = 6

        self.sanity_patterns = sn.all(
            [sn.assert_found('Finalising parallel run', self.stdout),
             sn.assert_found(r'^\s*[Ee]nd', self.stdout)])


class SurfaceFeatureExtract(OpenFOAMBaseTest):
    def __init__(self, **kwargs):
        super().__init__(
            'surfaceFeatureExtract',
            'OpenFOAM check of surfaceFeatureExtract: motorbike tutorial',
            **kwargs)

        self.sanity_patterns = sn.assert_found(r'^\s*[Ee]nd', self.stdout)


class TopoSet(OpenFOAMBaseTest):
    def __init__(self, **kwargs):
        super().__init__('topoSet',
                         'OpenFOAM check of topoSet: heatexchanger tutorial',
                         **kwargs)

        self.executable_opts = ['-region air', '-dict system/topoSetDict.1']
        self.sanity_patterns = sn.assert_found(r'^\s*[Ee]nd', self.stdout)


def _get_checks(**kwargs):
    return [FoamyHexMesh(**kwargs),
            BuoyantBoussinesqSimpleFoam(**kwargs),
            ChtMultiRegionSimpleFoam(**kwargs),
            InterMixingFoam(**kwargs),
            SimpleFoam(**kwargs),
            PimpleFoam(**kwargs),
            PotentialFoam(**kwargs),
            SnappyHexMesh(**kwargs),
            SetFields(**kwargs),
            BlockMesh(**kwargs),
            CheckMesh(**kwargs),
            CollapseEdges(**kwargs),
            CreateBaffles(**kwargs),
            DecomposePar(**kwargs),
            PatchSummary(**kwargs),
            ReconstructPar(**kwargs),
            ReconstructParMesh(**kwargs),
            SurfaceFeatureExtract(**kwargs),
            TopoSet(**kwargs)]
