# The first of the following tests verify the installation. The remaining
# tests operate on files. All netCDF files incl CDL metadata were
# downloaded from:
# https://www.unidata.ucar.edu/software/netcdf/examples/files.html
# To avoid large test files some of the originally downloaded files were split
# using 'cdo splitname <varname> <varname>_'.
# CDO has over 700 operators. To verify the basic functioning of CDO, I
# selected 'info' and 'merge'. The rationale of this selection is:
# - 'info' is probably the most basic operator and it verifies that metadata
# can be accessed correctly;
# - 'merge' is a rather complex operator which verifies that data and metadata
# can be read and written.
# The next step to enlarge the CDO verification would probably be to perform
# the test with different files having different structures and organisation.
# In fact, the test InfoNC4Test for example fails if the file is changed to
# 'test_hgroups.nc4'; it gives the error:
#   cdo info: Open failed on >test_hgroups.nc4<
#   Unsupported file structure"

import os

import reframe as rfm
import reframe.utility.sanity as sn
from reframe.core.pipeline import RunOnlyRegressionTest


class CDOBaseTest(rfm.RunOnlyRegressionTest):
    def __init__(self):
        super().__init__()
        self.sourcesdir = os.path.join(self.current_system.resourcesdir,
                                       'CDO-NCO')
        self.valid_systems = ['daint:gpu', 'daint:mc', 'dom:gpu', 'dom:mc',
                              'kesch:pn']
        if self.current_system.name == 'kesch':
            self.exclusive_access = True
            self.valid_prog_environs = ['PrgEnv-gnu-nompi']
            self.modules = ['cdo']
        else:
            self.valid_prog_environs = ['PrgEnv-gnu']
            self.modules = ['CDO']

        self.maintainers = ['SO']
        self.tags = {'production', 'mch'}


# Check that the netCDF loaded by the CDO module supports the nc4 filetype
# (nc4 support must be explicitly activated when the netCDF library is
# compiled...).
@rfm.simple_test
class CDO_DependencyTest(CDOBaseTest):
    def __init__(self):
        super().__init__()
        self.descr = ('verifies that the netCDF loaded by the CDO module '
                      'supports the nc4 filetype')
        self.sourcesdir = None
        self.executable = 'nc-config'
        self.executable_opts = ['--has-nc4']
        self.sanity_patterns = sn.assert_found(r'^yes', self.stdout)


@rfm.simple_test
class CDO_NC4SupportTest(CDOBaseTest):
    def __init__(self):
        super().__init__()
        self.descr = ('verifies that the CDO supports the nc4 filetype')
        self.sourcesdir = None
        self.executable = 'cdo'
        self.executable_opts = ['-V']
        self.sanity_patterns = sn.assert_found(r'^Filetypes:.*\snc4\s'
                                               r'nc4c\W', self.stderr)


# All CDO check load the CDO module (see CDOBaseTest). This test tries to load
# then the NCO module to see if there appear any conflicts. If there are no
# conflicts then self.stdout and self.stderr are empty. Note that the command
# 'module load NCO' cannot be passed via self.executable to srun as 'module'
# is not an executable. Thus, we run the command as a pre_run command and
# define as executable just an echo with no arguments.
@rfm.simple_test
class CDO_NCOModuleCompatibilityTest(CDOBaseTest):
    def __init__(self):
        super().__init__()
        self.descr = ('verifies compatibility with the NCO module')
        self.sourcesdir = None
        self.executable = 'echo'
        self.sanity_patterns = sn.assert_not_found(
            r'(?i)error|conflict|unsupported|failure', self.stderr)

    def setup(self, partition, environ, **job_opts):
        nco_name = 'nco' if self.current_system.name == 'kesch' else 'NCO'
        self.pre_run = ['module load %s' % nco_name]
        super().setup(partition, environ, **job_opts)


@rfm.simple_test
class CDO_InfoNCTest(CDOBaseTest):
    def __init__(self):
        super().__init__()
        self.descr = ('verifies reading info of a standard netCDF file')
        self.executable = 'cdo'
        self.executable_opts = ['info', 'sresa1b_ncar_ccsm3-example.nc']
        # TODO: Add here also Warning? then it fails currently...
        self.sanity_patterns = sn.all([
            sn.assert_not_found(r'(?i)unsupported|error', self.stderr),
            sn.assert_found(r'cdo info: Processed 688128 values from 5 '
                            r'variables over 1 timestep', self.stderr)
        ])


@rfm.simple_test
class CDO_InfoNC4Test(CDOBaseTest):
    def __init__(self):
        super().__init__()
        self.descr = ('verifies reading info of a netCDF-4 file')
        self.executable = 'cdo'
        self.executable_opts = [
            'info', 'test_echam_spectral-deflated_wind10_wl_ws.nc4']
        # TODO: fails currently with the file test_hgroups.nc4
        self.sanity_patterns = sn.all([
            sn.assert_not_found(r'(?i)unsupported|error', self.stderr),
            sn.assert_found(r'cdo info: Processed 442368 values from 3 '
                            r'variables over 8 timesteps', self.stderr)
        ])


@rfm.simple_test
class CDO_InfoNC4CTest(CDOBaseTest):
    def __init__(self):
        super().__init__()
        self.descr = ('verifies reading info of a compressed netCDF-4 file')
        self.executable = 'cdo'
        self.executable_opts = [
            'info', 'test_echam_spectral-deflated_wind10_wl_ws.nc4c']
        self.sanity_patterns = sn.all([
            sn.assert_not_found(r'(?i)unsupported|error', self.stderr),
            sn.assert_found(r'cdo info: Processed 442368 values from 3 '
                            r'variables over 8 timesteps', self.stderr)
        ])


@rfm.simple_test
class CDO_MergeNCTest(CDOBaseTest):
    def __init__(self):
        super().__init__()
        self.descr = ('verifies merging of 3 standard netCDF files')
        self.executable = 'cdo'
        self.executable_opts = [
            '-O', 'merge',
            'sresa1b_ncar_ccsm3-example_pr.nc',
            'sresa1b_ncar_ccsm3-example_tas.nc',
            'sresa1b_ncar_ccsm3-example_area.nc',
            'sresa1b_ncar_ccsm3-example_area_pr_tas_area.nc'
        ]
        self.sanity_patterns = sn.all([
            sn.assert_not_found(r'(?i)unsupported|error', self.stderr),
            sn.assert_found(r'cdo merge: Processed 98304 values from 3 '
                            r'variables over (1|3) timesteps?', self.stderr)
        ])


@rfm.simple_test
class CDO_MergeNC4Test(CDOBaseTest):
    def __init__(self):
        super().__init__()
        self.descr = ('verifies merging of 3 netCDF-4 files')
        self.executable = 'cdo'
        self.executable_opts = [
            '-O', 'merge',
            'test_echam_spectral-deflated_wind10.nc4',
            'test_echam_spectral-deflated_wl.nc4',
            'test_echam_spectral-deflated_ws.nc4',
            'test_echam_spectral-deflated_wind10_wl_ws.nc4'
        ]
        self.sanity_patterns = sn.all([
            sn.assert_not_found(r'(?i)unsupported|error', self.stderr),
            sn.assert_found(r'cdo merge: Processed 442368 values from 3 '
                            r'variables over (8|24) timesteps', self.stderr)
        ])


@rfm.simple_test
class CDO_MergeNC4CTest(CDOBaseTest):
    def __init__(self):
        super().__init__()
        self.descr = ('verifies merging and compressing of 3 compressed '
                      'netCDF-4 files')
        self.executable = 'cdo'
        self.executable_opts = [
            '-O', '-z', 'zip', 'merge',
            'test_echam_spectral-deflated_wind10.nc4c',
            'test_echam_spectral-deflated_wl.nc4c',
            'test_echam_spectral-deflated_ws.nc4c',
            'test_echam_spectral-deflated_wind10_wl_ws.nc4c'
        ]
        self.sanity_patterns = sn.all([
            sn.assert_not_found(r'(?i)unsupported|error', self.stderr),
            sn.assert_found(r'cdo merge: Processed 442368 values from 3 '
                            r'variables over (8|24) timesteps', self.stderr)
        ])
