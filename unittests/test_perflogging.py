# Copyright 2016-2024 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import contextlib
import io
import os
import pytest

import reframe as rfm
import reframe.core.logging as logging
import reframe.core.runtime as rt
import reframe.frontend.executors as executors
import reframe.utility.osext as osext
import reframe.utility.sanity as sn


class _MyPerfTest(rfm.RunOnlyRegressionTest):
    valid_systems = ['*']
    valid_prog_environs = ['*']
    executable = 'echo perf0=100 && echo perf1=50'
    reference = {
        '*': {
            'perf0': (100, -0.05, 0.05, 'unit0'),
            'perf1': (100, -0.05, 0.05, 'unit1')
        }
    }

    @sanity_function
    def validate(self):
        return sn.assert_found(r'perf0', self.stdout)

    @performance_function('unit0')
    def perf0(self):
        return sn.extractsingle(r'perf0=(\S+)', self.stdout, 1, float)

    @performance_function('unit1')
    def perf1(self):
        return sn.extractsingle(r'perf1=(\S+)', self.stdout, 1, float)


class _MyPerfParamTest(_MyPerfTest):
    p = parameter([1, 2])


class _MyFailingTest(rfm.RunOnlyRegressionTest):
    valid_systems = ['*']
    valid_prog_environs = ['*']
    executable = 'echo perf0=100'

    @sanity_function
    def validate(self):
        return False

    @performance_function('unit0')
    def perf0(self):
        return sn.extractsingle(r'perf0=(\S+)', self.stdout, 1, float)


class _LazyPerfTest(rfm.RunOnlyRegressionTest):
    valid_systems = ['*']
    valid_prog_environs = ['*']
    executable = 'echo perf0=100'

    @sanity_function
    def validate(self):
        return True

    @run_before('performance')
    def set_perf_vars(self):
        self.perf_variables = {
            'perf0': sn.make_performance_function(
                sn.extractsingle(r'perf0=(\S+)', self.stdout, 1, float),
                'unit0'
            )
        }


@pytest.fixture
def perf_test():
    return _MyPerfTest()


@pytest.fixture
def perf_param_tests():
    return [_MyPerfParamTest(variant_num=v)
            for v in range(_MyPerfParamTest.num_variants)]


@pytest.fixture
def failing_perf_test():
    return _MyFailingTest()


@pytest.fixture
def lazy_perf_test():
    return _LazyPerfTest()


@pytest.fixture
def simple_test():
    class _MySimpleTest(rfm.RunOnlyRegressionTest):
        valid_systems = ['*']
        valid_prog_environs = ['*']
        executable = 'echo hello'

        @sanity_function
        def validate(self):
            return sn.assert_found(r'hello', self.stdout)

    return _MySimpleTest()


@pytest.fixture
def config_perflog(make_config_file):
    def _config_perflog(fmt, perffmt=None, logging_opts=None):
        logging_config = {
            'level': 'debug2',
            'handlers': [{
                'type': 'stream',
                'name': 'stdout',
                'level': 'info',
                'format': '%(message)s'
            }],
            'handlers_perflog': [{
                'type': 'filelog',
                'prefix': '%(check_system)s/%(check_partition)s',
                'level': 'info',
                'format': fmt
            }]
        }
        if logging_opts:
            logging_config.update(logging_opts)

        if perffmt is not None:
            logging_config['handlers_perflog'][0]['format_perfvars'] = perffmt

        return make_config_file({'logging': [logging_config]})

    return _config_perflog


def _count_lines(filepath):
    count = 0
    with open(filepath) as fp:
        for line in fp:
            count += 1

    return count


def _assert_header(filepath, header):
    with open(filepath) as fp:
        assert fp.readline().strip() == header


def _assert_no_logging_error(fn, *args, **kwargs):
    captured_stderr = io.StringIO()
    with contextlib.redirect_stderr(captured_stderr):
        fn(*args, **kwargs)

    assert 'Logging error' not in captured_stderr.getvalue()


def test_perf_logging(make_runner, make_exec_ctx, perf_test,
                      config_perflog, tmp_path):
    make_exec_ctx(
        config_perflog(
            fmt=(
                '%(check_job_completion_time)s,%(version)s,'
                '%(check_display_name)s,%(check_system)s,'
                '%(check_partition)s,%(check_environ)s,'
                '%(check_jobid)s,%(check_result)s,%(check_perfvalues)s'
            ),
            perffmt=(
                '%(check_perf_value)s,%(check_perf_unit)s,'
                '%(check_perf_ref)s,%(check_perf_lower_thres)s,'
                '%(check_perf_upper_thres)s,%(check_perf_result)s,'
            )
        )
    )
    logging.configure_logging(rt.runtime().site_config)
    runner = make_runner()
    testcases = executors.generate_testcases([perf_test])
    runner.runall(testcases)

    logfile = tmp_path / 'perflogs' / 'generic' / 'default' / '_MyPerfTest.log'
    assert os.path.exists(logfile)
    assert _count_lines(logfile) == 2

    # Rerun with the same configuration and check that new entry is appended
    testcases = executors.generate_testcases([perf_test])
    runner = make_runner()
    _assert_no_logging_error(runner.runall, testcases)
    assert _count_lines(logfile) == 3

    # Change the configuration and rerun
    make_exec_ctx(
        config_perflog(
            fmt=(
                '%(check_job_completion_time)s,%(version)s,'
                '%(check_display_name)s,%(check_system)s,'
                '%(check_partition)s,%(check_environ)s,'
                '%(check_jobid)s,%(check_result)s,%(check_perfvalues)s'
            ),
            perffmt='%(check_perf_value)s,%(check_perf_unit)s,'
        )
    )
    logging.configure_logging(rt.runtime().site_config)
    testcases = executors.generate_testcases([perf_test])
    runner = make_runner()
    _assert_no_logging_error(runner.runall, testcases)
    assert _count_lines(logfile) == 2
    _assert_header(logfile,
                   'job_completion_time,version,display_name,system,partition,'
                   'environ,jobid,result,perf0_value,perf0_unit,'
                   'perf1_value,perf1_unit')

    logfile_prev = [(str(logfile) + '.h0', 3)]
    for f, num_lines in logfile_prev:
        assert os.path.exists(f)
        _count_lines(f) == num_lines

    # Change the test and rerun
    perf_test.perf_variables['perfN'] = perf_test.perf_variables['perf1']

    # We reconfigure the logging in order for the filelog handler to start
    # from a clean state
    logging.configure_logging(rt.runtime().site_config)
    testcases = executors.generate_testcases([perf_test])
    runner = make_runner()
    _assert_no_logging_error(runner.runall, testcases)
    assert _count_lines(logfile) == 2
    _assert_header(logfile,
                   'job_completion_time,version,display_name,system,partition,'
                   'environ,jobid,result,perf0_value,perf0_unit,'
                   'perf1_value,perf1_unit,perfN_value,perfN_unit')

    logfile_prev = [(str(logfile) + '.h0', 3), (str(logfile) + '.h1', 2)]
    for f, num_lines in logfile_prev:
        assert os.path.exists(f)
        _count_lines(f) == num_lines


def test_perf_logging_no_end_delim(make_runner, make_exec_ctx, perf_test,
                                   config_perflog, tmp_path):
    make_exec_ctx(
        config_perflog(
            fmt=(
                '%(check_job_completion_time)s,%(version)s,'
                '%(check_display_name)s,%(check_system)s,'
                '%(check_partition)s,%(check_environ)s,'
                '%(check_jobid)s,%(check_result)s,%(check_perfvalues)s'
            ),
            perffmt='%(check_perf_value)s,%(check_perf_unit)s'
        )
    )
    logging.configure_logging(rt.runtime().site_config)
    runner = make_runner()
    testcases = executors.generate_testcases([perf_test])
    _assert_no_logging_error(runner.runall, testcases)

    logfile = tmp_path / 'perflogs' / 'generic' / 'default' / '_MyPerfTest.log'
    assert os.path.exists(logfile)
    assert _count_lines(logfile) == 2

    with open(logfile) as fp:
        lines = fp.readlines()

    assert len(lines) == 2
    assert lines[0] == (
        'job_completion_time,version,display_name,system,partition,'
        'environ,jobid,result,perf0_value,perf0_unitperf1_value,perf1_unit\n'
    )
    assert '<error formatting the performance record' in lines[1]


def test_perf_logging_no_perfvars(make_runner, make_exec_ctx, perf_test,
                                  config_perflog, tmp_path):
    make_exec_ctx(
        config_perflog(
            fmt=(
                '%(check_job_completion_time)s,%(version)s,'
                '%(check_display_name)s,%(check_system)s,'
                '%(check_partition)s,%(check_environ)s,'
                '%(check_jobid)s,%(check_result)s,%(check_perfvalues)s'
            )
        )
    )
    logging.configure_logging(rt.runtime().site_config)
    runner = make_runner()
    testcases = executors.generate_testcases([perf_test])
    _assert_no_logging_error(runner.runall, testcases)

    logfile = tmp_path / 'perflogs' / 'generic' / 'default' / '_MyPerfTest.log'
    assert os.path.exists(logfile)
    assert _count_lines(logfile) == 2

    with open(logfile) as fp:
        lines = fp.readlines()

    assert len(lines) == 2
    assert lines[0] == (
        'job_completion_time,version,display_name,system,partition,'
        'environ,jobid,result,\n'
    )
    assert 'error' not in lines[1]


def test_perf_logging_multiline(make_runner, make_exec_ctx, perf_test,
                                simple_test, failing_perf_test,
                                config_perflog, tmp_path):
    make_exec_ctx(
        config_perflog(
            fmt=('%(check_job_completion_time)s|reframe %(version)s|'
                 '%(check_name)s|%(check_perf_var)s=%(check_perf_value)s|'
                 'ref=%(check_perf_ref)s (l=%(check_perf_lower_thres)s, '
                 'u=%(check_perf_upper_thres)s)|%(check_perf_unit)s|'
                 '%(check_perf_result)s'),
            logging_opts={'perflog_compat': True}
        )
    )
    logging.configure_logging(rt.runtime().site_config)
    runner = make_runner()
    testcases = executors.generate_testcases(
        [perf_test, simple_test, failing_perf_test]
    )
    _assert_no_logging_error(runner.runall, testcases)

    logfile = tmp_path / 'perflogs' / 'generic' / 'default' / '_MyPerfTest.log'
    assert os.path.exists(logfile)
    assert _count_lines(logfile) == 3

    # assert that the emitted lines are correct
    with open(logfile) as fp:
        lines = fp.readlines()

    version = osext.reframe_version()
    print(''.join(lines))
    assert lines[0] == ('job_completion_time|reframe version|name|'
                        'perf_var=perf_value|ref=perf_ref '
                        '(l=perf_lower_thres, u=perf_upper_thres)|perf_unit|'
                        'perf_result\n')
    assert lines[1].endswith(
        f'|reframe {version}|_MyPerfTest|'
        f'perf0=100.0|ref=100 (l=-0.05, u=0.05)|unit0|pass\n'
    )
    assert lines[2].endswith(
        f'|reframe {version}|_MyPerfTest|'
        f'perf1=50.0|ref=100 (l=-0.05, u=0.05)|unit1|fail\n'
    )


def test_perf_logging_lazy(make_runner, make_exec_ctx, lazy_perf_test,
                           config_perflog, tmp_path):
    make_exec_ctx(
        config_perflog(
            fmt=(
                '%(check_job_completion_time)s,%(version)s,'
                '%(check_display_name)s,%(check_system)s,'
                '%(check_partition)s,%(check_environ)s,'
                '%(check_jobid)s,%(check_result)s,%(check_perfvalues)s'
            ),
            perffmt=(
                '%(check_perf_value)s,%(check_perf_unit)s,'
                '%(check_perf_ref)s,%(check_perf_lower_thres)s,'
                '%(check_perf_upper_thres)s,'
            )
        )
    )
    logging.configure_logging(rt.runtime().site_config)
    runner = make_runner()
    testcases = executors.generate_testcases([lazy_perf_test])
    _assert_no_logging_error(runner.runall, testcases)

    logfile = (tmp_path / 'perflogs' / 'generic' / 'default' /
               '_LazyPerfTest.log')
    assert os.path.exists(logfile)


@pytest.fixture(params=['%(check_result)s|%(check_#ALL)s', '%(check_#ALL)s'])
def perflog_fmt(request):
    return request.param


def test_perf_logging_all_attrs(make_runner, make_exec_ctx, perf_test,
                                config_perflog, tmp_path, perflog_fmt):
    make_exec_ctx(config_perflog(fmt=perflog_fmt))
    logging.configure_logging(rt.runtime().site_config)
    runner = make_runner()
    testcases = executors.generate_testcases([perf_test])
    _assert_no_logging_error(runner.runall, testcases)

    logfile = tmp_path / 'perflogs' / 'generic' / 'default' / '_MyPerfTest.log'
    assert os.path.exists(logfile)
    with open(logfile) as fp:
        header = fp.readline()

    loggable_attrs = type(perf_test).loggable_attrs()
    assert (len(header.split('|')) ==
            len(loggable_attrs) + (perflog_fmt != '%(check_#ALL)s'))


def test_perf_logging_custom_vars(make_runner, make_exec_ctx,
                                  config_perflog, tmp_path):
    # Create two tests with different loggable variables
    class _X(_MyPerfTest):
        x = variable(int, value=1, loggable=True)

    class _Y(_MyPerfTest):
        y = variable(int, value=2, loggable=True)

    make_exec_ctx(config_perflog(fmt='%(check_result)s|%(check_#ALL)s'))
    logging.configure_logging(rt.runtime().site_config)
    runner = make_runner()
    testcases = executors.generate_testcases([_X(), _Y()])
    _assert_no_logging_error(runner.runall, testcases)

    logfiles = [tmp_path / 'perflogs' / 'generic' / 'default' / '_X.log',
                tmp_path / 'perflogs' / 'generic' / 'default' / '_Y.log']
    for f in logfiles:
        with open(f) as fp:
            header = fp.readline().strip()
            if os.path.basename(f).startswith('_X'):
                assert 'x' in header.split('|')
            else:
                assert 'y' in header.split('|')


def test_perf_logging_param_test(make_runner, make_exec_ctx, perf_param_tests,
                                 config_perflog, tmp_path):
    make_exec_ctx(config_perflog(fmt='%(check_result)s|%(check_#ALL)s'))
    logging.configure_logging(rt.runtime().site_config)
    runner = make_runner()
    testcases = executors.generate_testcases(perf_param_tests)
    _assert_no_logging_error(runner.runall, testcases)

    logfile = (tmp_path / 'perflogs' / 'generic' /
               'default' / '_MyPerfParamTest.log')
    assert os.path.exists(logfile)
    assert _count_lines(logfile) == 3


def test_perf_logging_sanity_failure(make_runner, make_exec_ctx,
                                     config_perflog, tmp_path):
    class _X(_MyPerfTest):
        @sanity_function
        def validate(self):
            return sn.assert_true(0, msg='no way')

    make_exec_ctx(config_perflog(
        fmt='%(check_result)s|%(check_fail_reason)s|%(check_perfvalues)s',
        perffmt='%(check_perf_value)s|'
    ))
    logging.configure_logging(rt.runtime().site_config)
    runner = make_runner()
    testcases = executors.generate_testcases([_X()])
    _assert_no_logging_error(runner.runall, testcases)

    logfile = tmp_path / 'perflogs' / 'generic' / 'default' / '_X.log'
    assert os.path.exists(logfile)
    with open(logfile) as fp:
        lines = fp.readlines()

    assert len(lines) == 2
    assert lines[1] == 'fail|sanity error: no way|None|None\n'
