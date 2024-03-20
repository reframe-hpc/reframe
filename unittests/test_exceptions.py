# Copyright 2016-2024 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import pytest

import reframe.core.exceptions as exc


def assert_args(exc_type, *args):
    e = exc_type(*args)
    assert args == e.args


def test_soft_error():
    with pytest.raises(exc.ReframeError, match=r'random error'):
        raise exc.ReframeError('random error')

    assert_args(exc.ReframeError, 'error msg')
    assert_args(exc.ReframeError, 'error msg', 'another arg')


def test_reraise_soft_error():
    try:
        try:
            raise ValueError('random value error')
        except ValueError as e:
            # reraise as ReframeError
            raise exc.ReframeError('soft error') from e
    except exc.ReframeError as e:
        assert 'soft error: random value error' == str(e)


def test_fatal_error():
    try:
        raise exc.ReframeFatalError('fatal error')
    except Exception:
        pytest.fail('fatal error should not derive from Exception')
    except BaseException:
        pass


def test_reraise_fatal_error():
    try:
        try:
            raise ValueError('random value error')
        except ValueError as e:
            # reraise as ReframeError
            raise exc.ReframeFatalError('fatal error') from e
    except exc.ReframeFatalError as e:
        assert 'fatal error: random value error' == str(e)


def test_spawned_process_error():
    exc_args = ('foo bar', 'partial output', 'error message', 1)
    e = exc.SpawnedProcessError(*exc_args)
    with pytest.raises(
            exc.ReframeError,
            match=(r"command 'foo bar' failed with exit code 1:\n"
                   r"--- stdout ---\n"
                   r'partial output\n'
                   r"--- stdout ---\n"
                   r"--- stderr ---\n"
                   r"error message\n"
                   r"--- stderr ---")
    ):
        raise e

    assert exc_args == e.args


def test_spawned_process_error_list_args():
    exc_args = (['foo', 'bar'], 'partial output', 'error message', 1)
    e = exc.SpawnedProcessError(*exc_args)
    with pytest.raises(
            exc.ReframeError,
            match=(r"command 'foo bar' failed with exit code 1:\n"
                   r"--- stdout ---\n"
                   r'partial output\n'
                   r"--- stdout ---\n"
                   r"--- stderr ---\n"
                   r"error message\n"
                   r"--- stderr ---")
    ):
        raise e

    assert exc_args == e.args


def test_spawned_process_error_nostdout():
    exc_args = ('foo bar', '', 'error message', 1)
    e = exc.SpawnedProcessError(*exc_args)
    with pytest.raises(
            exc.ReframeError,
            match=(r"command 'foo bar' failed with exit code 1:\n"
                   r"--- stdout ---\n"
                   r"--- stdout ---\n"
                   r"--- stderr ---\n"
                   r"error message\n"
                   r"--- stderr ---")
    ):
        raise e


def test_spawned_process_error_nostderr():
    exc_args = ('foo bar', 'partial output', '', 1)
    e = exc.SpawnedProcessError(*exc_args)
    with pytest.raises(
            exc.ReframeError,
            match=(r"command 'foo bar' failed with exit code 1:\n"
                   r"--- stdout ---\n"
                   r'partial output\n'
                   r"--- stdout ---\n"
                   r"--- stderr ---\n"
                   r"--- stderr ---")
    ):
        raise e


def test_spawned_process_timeout():
    exc_args = ('foo bar', 'partial output', 'partial error', 10)
    e = exc.SpawnedProcessTimeout(*exc_args)
    with pytest.raises(exc.ReframeError,
                       match=(r"command 'foo bar' timed out after 10s:\n"
                              r"--- stdout ---\n"
                              r'partial output\n'
                              r"--- stdout ---\n"
                              r"--- stderr ---\n"
                              r"partial error\n"
                              r"--- stderr ---")):
        raise e


def test_spawned_process_timeout_nostdout():
    exc_args = ('foo bar', '', 'partial error', 10)
    e = exc.SpawnedProcessTimeout(*exc_args)
    with pytest.raises(exc.ReframeError,
                       match=(r"command 'foo bar' timed out after 10s:\n"
                              r"--- stdout ---\n"
                              r"--- stdout ---\n"
                              r"--- stderr ---\n"
                              r"partial error\n"
                              r"--- stderr ---")):
        raise e


def test_spawned_process_timeout_nostderr():
    exc_args = ('foo bar', 'partial output', '', 10)
    e = exc.SpawnedProcessTimeout(*exc_args)
    with pytest.raises(exc.ReframeError,
                       match=(r"command 'foo bar' timed out after 10s:\n"
                              r"--- stdout ---\n"
                              r'partial output\n'
                              r"--- stdout ---\n"
                              r"--- stderr ---\n"
                              r"--- stderr ---")):
        raise e


def test_job_error():
    exc_args = ('some error',)
    e = exc.JobError(*exc_args, jobid=1234)
    assert 1234 == e.jobid
    with pytest.raises(exc.JobError, match=r'\[jobid=1234\] some error'):
        raise e

    assert exc_args == e.args


def test_reraise_job_error():
    try:
        try:
            raise ValueError('random value error')
        except ValueError as e:
            raise exc.JobError('some error', jobid=1234) from e
    except exc.JobError as e:
        assert '[jobid=1234] some error: random value error' == str(e)


def test_reraise_job_error_no_message():
    try:
        try:
            raise ValueError('random value error')
        except ValueError as e:
            raise exc.JobError(jobid=1234) from e
    except exc.JobError as e:
        assert '[jobid=1234]: random value error' == str(e)
