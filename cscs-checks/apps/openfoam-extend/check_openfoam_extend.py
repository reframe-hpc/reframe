import os

import reframe.utility.sanity as sn
from reframe.core.pipeline import RunOnlyRegressionTest


class OpenfoamExtendBaseTest(RunOnlyRegressionTest):
    def __init__(self, check_name, check_descr, **kwargs):
        super().__init__('OpenfoamExtend_%s' % (check_name),
                         os.path.dirname(__file__), **kwargs)

        self.descr      = check_descr
        self.executable = check_name
        self.sourcesdir = os.path.join(self.current_system.resourcesdir,
                                       'OpenFOAM-Extend', check_name)

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

        self.maintainers = ['MaKra']
        self.tags = {'scs', 'production'}

    def setup(self, partition, environ, **job_opts):
        super().setup(partition, environ, **job_opts)
        self.job.pre_run = [
            'source $FOAM_INST_DIR/foam-extend-4.0/etc/bashrc']


class BlockMesh(OpenfoamExtendBaseTest):
    def __init__(self, **kwargs):
        super().__init__('blockMesh',
                         'OpenFOAM-Extend blockMesh from '
                         'the dambreak tutorial',
                         **kwargs)


class SnappyHexMesh(OpenfoamExtendBaseTest):
    def __init__(self, **kwargs):
        super().__init__(
            'snappyHexMesh',
            'OpenFOAM-Extend check of snappyHexMesh: motorbike tutorial',
            **kwargs)


class SimpleFoam(OpenfoamExtendBaseTest):
    def __init__(self, **kwargs):
        super().__init__(
            'simpleFoam',
            'OpenFOAM-Extend check of simpleFoam: motorbike tutorial',
            **kwargs)

        self.executable_opts = ['-parallel']
        self.num_tasks = 6
        self.num_tasks_per_node = 6

        result = sn.extractall(
            r'time step continuity errors : '
            r'\S+\s\S+ = \S+\sglobal = (?P<res>-?\S+),',
            self.stdout, 'res', float)
        self.sanity_patterns = sn.all(
            sn.map(lambda x: sn.assert_lt(abs(x), 5.e-04), result))


class SetFields(OpenfoamExtendBaseTest):
    def __init__(self, **kwargs):
        super().__init__(
            'setFields',
            'OpenFOA-Extend  check of setFields: dambreak tutorial',
            **kwargs)


class InterMixingFoam(OpenfoamExtendBaseTest):
    def __init__(self, **kwargs):
        super().__init__(
            'interMixingFoam',
            'OpenFOA-Extend  check of interMixingFoam: dambreak tutorial',
            **kwargs)

        self.sanity_patterns = sn.assert_eq(sn.count(sn.findall(
            r'Air phase volume fraction', self.stdout)), 2944)


class BuoyantBoussinesqSimpleFoam(OpenfoamExtendBaseTest):
    def __init__(self, **kwargs):
        super().__init__(
            'buoyantBoussinesqSimpleFoam',
            'OpenFOAM-Extend check buoyantBoussinesqSimpleFoam: hotRoom test',
            **kwargs)

        self.executable = 'buoyantBoussinesqSimpleFoam'

        result = sn.extractall(
            r'\sglobal\s=\s(?P<res>\S+),',
            self.stdout, 'res', float)
        self.sanity_patterns = sn.all(
            sn.map(lambda x: sn.assert_lt(abs(x), 1.e-17), result))


class LaplacianFoam(OpenfoamExtendBaseTest):
    def __init__(self, **kwargs):
        super().__init__('laplacianFoam',
                         'OpenFOAM-Extend check of setFields: flange tutorial',
                         **kwargs)


class FoamToEnsight(OpenfoamExtendBaseTest):
    def __init__(self, **kwargs):
        super().__init__('foamToEnsight',
                         'OpenFOAM-Extend check of setFields: flange tutorial',
                         **kwargs)


class SetSet(OpenfoamExtendBaseTest):
    def __init__(self, **kwargs):
        super().__init__(
            'setSet',
            'OpenFOAM-Extend check of setFields: multi region heater tutorial',
            **kwargs)

        self.executable_opts  = ['-batch makeCellSets.setSet']


class SetsToZones(OpenfoamExtendBaseTest):
    def __init__(self, **kwargs):
        super().__init__(
            'setsToZones',
            'OpenFOAM-Extend check of setFields: multi region heater tutorial',
            **kwargs)

        self.executable_opts  = ['-noFlipMap']


class SplitMeshRegions(OpenfoamExtendBaseTest):
    def __init__(self, **kwargs):
        super().__init__(
            'splitMeshRegions',
            'OpenFOAM-Extend check of setFields: multi region heater tutorial',
            **kwargs)

        self.executable_opts  = ['-cellZones', '-overwrite']


class DecomposePar(OpenfoamExtendBaseTest):
    def __init__(self, **kwargs):
        super().__init__(
            'decomposePar',
            'OpenFOAM-Extend check of reconstructPar: multiRegionHeater test',
            **kwargs)

        self.executable_opts  = ['-region heater']


class ChtMultiRegionSimpleFoam(OpenfoamExtendBaseTest):
    def __init__(self, **kwargs):
        super().__init__(
            'chtMultiRegionSimpleFoam',
            'OpenFOAM-Extend check of reconstructPar: multiRegionHeater test',
            **kwargs)

        self.executable_opts = ['-parallel']
        self.num_tasks          = 4
        self.num_tasks_per_node = 4

        result = sn.extractall(
            r'\sglobal\s=\s(?P<res>-?\S+),',
            self.stdout, 'res', float)[-5:]
        self.sanity_patterns = sn.all(
            sn.map(lambda x: sn.assert_lt(abs(x), 1.e-04), result))


class ReconstructPar(OpenfoamExtendBaseTest):
    def __init__(self, **kwargs):
        super().__init__(
            'reconstructPar',
            'OpenFOAM-Extend check of reconstructPar: multiRegionHeater test',
            **kwargs)

        self.executable_opts = ['-latestTime', '-region heater']
        self.readonly_files  = ['processor0',
                                'processor1',
                                'processor2',
                                'processor3']

        self.sanity_patterns = sn.assert_found(r'Time = 1000', self.stdout)


def _get_checks(**kwargs):
    return [
        BlockMesh(**kwargs),
        SnappyHexMesh(**kwargs),
        SimpleFoam(**kwargs),
        SetFields(**kwargs),
        InterMixingFoam(**kwargs),
        BuoyantBoussinesqSimpleFoam(**kwargs),
        LaplacianFoam(**kwargs),
        FoamToEnsight(**kwargs),
        SetSet(**kwargs),
        SetsToZones(**kwargs),
        SplitMeshRegions(**kwargs),
        DecomposePar(**kwargs),
        ChtMultiRegionSimpleFoam(**kwargs),
        ReconstructPar(**kwargs)]
