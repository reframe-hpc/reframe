# The first of the following tests verify the installation. The remaining
# tests operate on files. All netCDF files incl CDL metadata were
# downloaded from:
# https://www.unidata.ucar.edu/software/netcdf/examples/files.html
# To avoid large test files some of the originally downloaded files were split
# using 'cdo splitname <varname> <varname>_'.
# NCO permits to do a large amount of operations. To verify the basic
# functioning of NCO, I selected the operations that are equivivalent of
# those chosen for the CDO tests (the rationale of this selection is explained
# in the corresponding file):
# - 'ncks --trd -M' ("print metadata to screen") corresponds to 'cdo info'
# - 'ncks -A' ("append") corresponds to 'cdo merge'

import os
import reframe.utility.sanity as sn

from reframe.core.pipeline import RunOnlyRegressionTest


class NCOBaseCheck(RunOnlyRegressionTest):
    def __init__(self, sub_check, **kwargs):
        super().__init__('NCO_' + sub_check + '_check',
                         os.path.dirname(__file__), **kwargs)
        self.sourcesdir = os.path.join(self.current_system.resourcesdir,
                                       'CDO-NCO')
        self.valid_systems = ['daint:gpu', 'daint:mc', 'dom:gpu', 'dom:mc',
                              'kesch:pn', 'kesch:cn']
        self.valid_prog_environs = ['PrgEnv-gnu']
        self.maintainers = ['SO']
        self.tags = {'production'}

    def setup(self, partition, environ, **job_opts):
        nco_name = 'nco' if self.current_system.name == 'kesch' else 'NCO'
        self.modules = [nco_name]
        super().setup(partition, environ, **job_opts)


# Check that the netCDF loaded by the NCO module supports the nc4 filetype
# (nc4 support must be explicitly activated when the netCDF library is
# compiled...).
class DependencyCheck(NCOBaseCheck):
    def __init__(self, **kwargs):
        super().__init__('dependency', **kwargs)
        self.descr = ('verifies that the netCDF loaded by the NCO module '
                      'supports the nc4 filetype')
        self.sourcesdir = None
        self.executable = 'nc-config'
        self.executable_opts = ['--has-nc4']
        self.sanity_patterns = sn.assert_found(r'^yes', self.stdout)


class NC4SupportCheck(NCOBaseCheck):
    def __init__(self, **kwargs):
        super().__init__('nc4_support', **kwargs)
        self.descr = ('verifies that the NCO supports the nc4 filetype')
        self.sourcesdir = None
        self.executable = 'ncks'
        self.executable_opts = ['-r']
        self.sanity_patterns = sn.all([
            sn.assert_found(r'^netCDF4/HDF5 available\s+Yes\W', self.stdout),
            sn.assert_found(r'^netCDF4/HDF5 enabled\s+Yes\W', self.stdout)
        ])


# All NCO check load the NCO module (see NCOBaseCheck). This test tries to load
# then the CDO module to see if there appear any conflicts. If there are no
# conflicts then self.stdout and self.stderr are empty. Note that the command
# 'module load CDO' cannot be passed via self.executable to srun as 'module'
# is not an executable. Thus, we run the command as a pre_run command and
# define as executable just an echo with no arguments.
class CDOModuleCompatibilityCheck(NCOBaseCheck):
    def __init__(self, **kwargs):
        super().__init__('cdo_module_compat', **kwargs)
        self.descr = ('verifies compatibility with the CDO module')
        self.sourcesdir = None
        output_file = 'nco_cdo_check.out'
        self.executable = 'echo > %s' % output_file
        self.sanity_patterns = sn.all([
            sn.assert_not_found(r'.+', output_file),
            sn.assert_not_found(r'.+', self.stderr)
        ])

    def setup(self, partition, environ, **job_opts):
        cdo_name = 'cdo' if self.current_system.name == 'kesch' else 'CDO'
        self.pre_run = ['module load %s' % cdo_name]
        super().setup(partition, environ, **job_opts)


class InfoNCCheck(NCOBaseCheck):
    def __init__(self, **kwargs):
        super().__init__('info_nc', **kwargs)
        self.descr = ('verifies reading info of a standard netCDF file')
        self.executable = 'ncks'
        self.executable_opts = ['-M', 'sresa1b_ncar_ccsm3-example.nc']
        self.sanity_patterns = sn.all([
            sn.assert_not_found(r'(?i)unsupported|error', self.stderr),
            sn.assert_found(r'^Global attribute \d+: model_name_english',
                            self.stdout)
        ])


class InfoNC4Check(NCOBaseCheck):
    def __init__(self, **kwargs):
        super().__init__('info_nc4', **kwargs)
        self.descr = ('verifies reading info of a netCDF-4 file')
        self.executable = 'ncks'
        self.executable_opts = [
            '-M', 'test_echam_spectral-deflated_wind10_wl_ws.nc4'
        ]
        self.sanity_patterns = sn.all([
            sn.assert_not_found(r'(?i)unsupported|error', self.stderr),
            sn.assert_found(r'^Global attribute \d+: CDO, size = 63 NC_CHAR',
                            self.stdout)
        ])


class InfoNC4CCheck(NCOBaseCheck):
    def __init__(self, **kwargs):
        super().__init__('info_nc4c', **kwargs)
        self.descr = ('verifies reading info of a compressed netCDF-4 file')
        self.executable = 'ncks'
        self.executable_opts = [
            '-M', 'test_echam_spectral-deflated_wind10_wl_ws.nc4c'
        ]
        self.sanity_patterns = sn.all([
            sn.assert_not_found(r'(?i)unsupported|(?i)error', self.stderr),
            sn.assert_found(r'^Global attribute \d+: CDO, size = 63 NC_CHAR',
                            self.stdout)
        ])


class MergeNCCheck(NCOBaseCheck):
    def __init__(self, **kwargs):
        super().__init__('merge_nc', **kwargs)
        self.descr = ('verifies merging of two standard netCDF files')
        self.executable = 'ncks'
        self.executable_opts = ['-A',
                                'sresa1b_ncar_ccsm3-example_pr.nc',
                                'sresa1b_ncar_ccsm3-example_tas.nc']
        # NOTE: we only verify that there is no error produced. We do not
        # verify the resulting file. Verifying would not be straight forward
        # as the following does not work (it seems that there is a timestamp
        # of file creation inserted in the metadata or similar):
        # diff sresa1b_ncar_ccsm3-example_tas.nc \
        #      sresa1b_ncar_ccsm3-example_pr_tas.nc
        self.sanity_patterns = sn.all([
            sn.assert_not_found(r'(?i)unsupported|error', self.stderr),
            sn.assert_not_found(r'(?i)unsupported|error', self.stdout)
        ])


class MergeNC4Check(NCOBaseCheck):
    def __init__(self, **kwargs):
        super().__init__('merge_nc4', **kwargs)
        self.descr = ('verifies merging of two netCDF-4 files')
        self.executable = 'ncks'
        self.executable_opts = ['-A',
                                'test_echam_spectral-deflated_wind10.nc4',
                                'test_echam_spectral-deflated_wl.nc4']
        # NOTE: we only verify that there is no error produced. We do not
        # verify the resulting file. Verifying would not be straight forward
        # as the following does not work (it seems that there is a timestamp
        # of file creation inserted in the metadata or similar):
        # diff test_echam_spectral - deflated_wl.nc4 \
        #      test_echam_spectral - deflated_wind10_wl.nc4
        self.sanity_patterns = sn.all([
            sn.assert_not_found(r'(?i)unsupported|error', self.stderr),
            sn.assert_not_found(r'(?i)unsupported|error', self.stdout)
        ])


class MergeNC4CCheck(NCOBaseCheck):
    def __init__(self, **kwargs):
        super().__init__('merge_nc4c', **kwargs)
        self.descr = ('verifies merging and compressing of two compressed '
                      'netCDF-4 files')
        self.executable = 'ncks'
        self.executable_opts = ['-A', '-L', '2',
                                'test_echam_spectral-deflated_wind10.nc4c',
                                'test_echam_spectral-deflated_wl.nc4c']
        # NOTE: we only verify that there is no error produced. We do not
        # verify the resulting file. Verifying would not be straight forward
        # as the following does not work (it seems that there is a timestamp
        # of file creation inserted in the metadata or similar):
        # diff test_echam_spectral-deflated_wl.nc4c \
        #      test_echam_spectral-deflated_wind10_wl.nc4c
        self.sanity_patterns = sn.all([
            sn.assert_not_found(r'(?i)unsupported|error', self.stderr),
            sn.assert_not_found(r'(?i)unsupported|error', self.stdout)
        ])


def _get_checks(**kwargs):
    return [
        DependencyCheck(**kwargs), NC4SupportCheck(**kwargs),
        CDOModuleCompatibilityCheck(**kwargs), InfoNCCheck(**kwargs),
        InfoNC4Check(**kwargs), InfoNC4CCheck(**kwargs),
        MergeNCCheck(**kwargs), MergeNC4Check(**kwargs),
        MergeNC4CCheck(**kwargs)
    ]
