import pytest
import lmod_helper

# On Cheyenne, output of `module reset; module rm intel; module avail gnu`
ml_av_out_easy = """
Resetting modules to system default

Activating Modules:
  1) ncarcompilers/0.5.0     2) netcdf/4.7.3     3) openmpi/3.1.4

Inactive Modules:
  1) ncarcompilers/0.5.0     2) netcdf/4.7.3     3) openmpi/3.1.4


--------------------------------------------------------------------------------------------- /glade/u/apps/dav/modulefiles/default/compilers ---------------------------------------------------------------------------------------------
   gnu/6.4.0    gnu/7.3.0    gnu/7.4.0    gnu/8.3.0 (D)    gnu/9.1.0

----------------------------------------------------------------------------------------------- /glade/u/apps/dav/modulefiles/default/idep ------------------------------------------------------------------------------------------------
   gnuplot/5.2.2

  Where:
   D:  Default Module

Use "module spider" to find all possible modules.
Use "module keyword key1 key2 ..." to search for all possible modules matching any of the "keys".

"""

# On Cheyenne, output of `module reset; module rm intel; module load gnu/7.3.0; module avail netcdf`
ml_av_out_tricky = """
Resetting modules to system default

Activating Modules:
  1) ncarcompilers/0.5.0     2) netcdf/4.7.3     3) openmpi/3.1.4


Inactive Modules:
  1) ncarcompilers/0.5.0     2) netcdf/4.7.3     3) openmpi/3.1.4


Activating Modules:
  1) openmpi/3.1.4


--------------------------------------------------------------------------------------------- /glade/u/apps/dav/modulefiles/default/gnu/7.3.0 ---------------------------------------------------------------------------------------------
   netcdf/4.6.0    netcdf/4.6.1 (D)

  Where:
   D:  Default Module

Use "module spider" to find all possible modules.
Use "module keyword key1 key2 ..." to search for all possible modules matching any of the "keys".

"""

class PopenCommunicateMock():
    def __init__(self, output, always=True):
        self.output = output
        self.always = always

    def exec(self, command):
        if self.always:
            return self.output
        else:
            return self.output[command]

def test_all_avail_mods():
    expected = ['6.4.0', '7.3.0', '7.4.0', '8.3.0', '9.1.0']
    assert expected == lmod_helper._all_avail_mods("gnu", ml_av_out_easy)

def test_all_avail_mods_tricky():
    expected = ['netcdf/4.6.0', 'netcdf/4.6.1']
    assert expected == lmod_helper._all_avail_mods("netcdf", ml_av_out_tricky)

@pytest.mark.skip("automatically tested with _expand_module_set")
def test_loop_among_one():
    assert False

def test_expand_module_set_simple():
    module_glob = 'module load gnu/*'
    module_set  = ['module reset; module rm intel']
    expected =    ['module reset; module rm intel; module load gnu/6.4.0',
                   'module reset; module rm intel; module load gnu/7.3.0',
                   'module reset; module rm intel; module load gnu/7.4.0',
                   'module reset; module rm intel; module load gnu/8.3.0',
                   'module reset; module rm intel; module load gnu/9.1.0']
    pc = PopenCommunicateMock(ml_av_out_easy)
    assert expected == lmod_helper._expand_module_set(module_set, module_glob, pc)

def test_expand_module_set_nested():
    module_glob = 'module load netcdf/*'
    module_set  = ['module load gnu/6.4.0',
                   'module load gnu/7.3.0',
                   'module load gnu/7.4.0']

    mock_rules = { # keys must match the module set
                   'module load gnu/6.4.0; module avail netcdf': 'netcdf/4.7.3',
                   'module load gnu/7.3.0; module avail netcdf': 'netcdf/4.7.3 netcdf/4.6.0 netcdf/4.6.1',
                   'module load gnu/7.4.0; module avail netcdf': 'netcdf/4.7.1 netcdf/4.6.0 netcdf/4.6.3' }

    expected =    ['module load gnu/6.4.0; module load netcdf/4.7.3',

                   'module load gnu/7.3.0; module load netcdf/4.7.3',
                   'module load gnu/7.3.0; module load netcdf/4.6.0',
                   'module load gnu/7.3.0; module load netcdf/4.6.1',

                   'module load gnu/7.4.0; module load netcdf/4.7.1',
                   'module load gnu/7.4.0; module load netcdf/4.6.0',
                   'module load gnu/7.4.0; module load netcdf/4.6.3' ]
    pc = PopenCommunicateMock(mock_rules, always = False)
    assert expected == lmod_helper._expand_module_set(module_set, module_glob, pc)

def test_nothing_to_do():
    no_glob = "module reset; module sw intel gnu; module load netcdf/1.2.3; module load python/4.5.6"
    results = lmod_helper.loop_among_all(no_glob)
    assert results == [no_glob]

def test_loop_among_all_in_the_middle():
    pc = PopenCommunicateMock(ml_av_out_easy)
    result = lmod_helper.loop_among_all("module reset; module sw intel gnu/*; module load netcdf/1.2.3; module load python/4.5.6", pc)
    expected_versions = ['6.4.0', '7.3.0', '7.4.0', '8.3.0', '9.1.0']
    expected = ["module reset; module sw intel gnu/" + i + "; module load netcdf/1.2.3; module load python/4.5.6" for i in expected_versions]
    assert expected == result

# partially tested by test_expand_module_set_nested, here checking that the things are put together correctly
def test_loop_among_all_two_globs():
    mock_rules = { 'ml reset; ml rm intel; module avail gnu':                  'gnu/6.4.0 gnu/7.3.0 gnu/7.4.0',
                   'ml reset; ml rm intel; ml gnu/6.4.0; module avail netcdf': 'netcdf/4.7.3',
                   'ml reset; ml rm intel; ml gnu/7.3.0; module avail netcdf': 'netcdf/4.7.3 netcdf/4.6.0 netcdf/4.6.1',
                   'ml reset; ml rm intel; ml gnu/7.4.0; module avail netcdf': 'netcdf/4.7.1 netcdf/4.6.0 netcdf/4.6.3' }
    pc = PopenCommunicateMock(mock_rules, always = False)
    results = lmod_helper.loop_among_all("ml reset; ml rm intel; ml gnu/*; ml netcdf/*", pc)
    expected =    ['ml reset; ml rm intel; ml gnu/6.4.0; ml netcdf/4.7.3',

                   'ml reset; ml rm intel; ml gnu/7.3.0; ml netcdf/4.7.3',
                   'ml reset; ml rm intel; ml gnu/7.3.0; ml netcdf/4.6.0',
                   'ml reset; ml rm intel; ml gnu/7.3.0; ml netcdf/4.6.1',

                   'ml reset; ml rm intel; ml gnu/7.4.0; ml netcdf/4.7.1',
                   'ml reset; ml rm intel; ml gnu/7.4.0; ml netcdf/4.6.0',
                   'ml reset; ml rm intel; ml gnu/7.4.0; ml netcdf/4.6.3' ]
    assert expected == results
