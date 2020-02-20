import pytest
import unittest

import reframe.core.exceptions as exc


def raise_exc(exc):
    raise exc


class TestExceptions(unittest.TestCase):
    def assert_args(self, exc_type, *args):
        e = exc_type(*args)
        assert args == e.args

    def test_soft_error(self):
        with pytest.raises(exc.ReframeError, match=r'random error'):
            raise_exc(exc.ReframeError('random error'))
        self.assert_args(exc.ReframeError, 'error msg')
        self.assert_args(exc.ReframeError, 'error msg', 'another arg')

    def test_reraise_soft_error(self):
        try:
            try:
                raise ValueError('random value error')
            except ValueError as e:
                # reraise as ReframeError
                raise exc.ReframeError('soft error') from e
        except exc.ReframeError as e:
            assert 'soft error: random value error' == str(e)

    def test_fatal_error(self):
        try:
            raise exc.ReframeFatalError('fatal error')
        except Exception:
            pytest.fail('fatal error should not derive from Exception')
        except BaseException:
            pass

    def test_reraise_fatal_error(self):
        try:
            try:
                raise ValueError('random value error')
            except ValueError as e:
                # reraise as ReframeError
                raise exc.ReframeFatalError('fatal error') from e
        except exc.ReframeFatalError as e:
            assert 'fatal error: random value error' == str(e)

    def test_spawned_process_error(self):
        exc_args = ('foo bar', 'partial output', 'error message', 1)
        e = exc.SpawnedProcessError(*exc_args)
        with pytest.raises(
            exc.ReframeError,
            match=r"command 'foo bar' failed with exit code 1:\n"
                  r"=== STDOUT ===\n"
                  r'partial output\n'
                  r"=== STDERR ===\n"
                  r"error message"):
            raise_exc(e)
        assert exc_args == e.args

    def test_spawned_process_error_nostdout(self):
        exc_args = ('foo bar', '', 'error message', 1)
        e = exc.SpawnedProcessError(*exc_args)
        with pytest.raises(
            exc.ReframeError,
            match=r"command 'foo bar' failed with exit code 1:\n"
                  r"=== STDOUT ===\n"
                  r"=== STDERR ===\n"
                  r"error message"):
            raise_exc(e)

    def test_spawned_process_error_nostderr(self):
        exc_args = ('foo bar', 'partial output', '', 1)
        e = exc.SpawnedProcessError(*exc_args)
        with pytest.raises(
            exc.ReframeError,
            match=r"command 'foo bar' failed with exit code 1:\n"
                  r"=== STDOUT ===\n"
                  r'partial output\n'
                  r"=== STDERR ==="):
            raise_exc(e)

    def test_spawned_process_timeout(self):
        exc_args = ('foo bar', 'partial output', 'partial error', 10)
        e = exc.SpawnedProcessTimeout(*exc_args)
        with pytest.raises(exc.ReframeError,
                           match=r"command 'foo bar' timed out after 10s:\n"
                                 r"=== STDOUT ===\n"
                                 r'partial output\n'
                                 r"=== STDERR ===\n"
                                 r"partial error"):
            raise_exc(e)
        assert exc_args == e.args

    def test_spawned_process_timeout_nostdout(self):
        exc_args = ('foo bar', '', 'partial error', 10)
        e = exc.SpawnedProcessTimeout(*exc_args)
        with pytest.raises(exc.ReframeError,
                           match=r"command 'foo bar' timed out after 10s:\n"
                                 r"=== STDOUT ===\n"
                                 r"=== STDERR ===\n"
                                 r"partial error"):
            raise_exc(e)

    def test_spawned_process_timeout_nostderr(self):
        exc_args = ('foo bar', 'partial output', '', 10)
        e = exc.SpawnedProcessTimeout(*exc_args)
        with pytest.raises(exc.ReframeError,
                           match=r"command 'foo bar' timed out after 10s:\n"
                                 r"=== STDOUT ===\n"
                                 r'partial output\n'
                                 r"=== STDERR ==="):
            raise_exc(e)

    def test_job_error(self):
        exc_args = ('some error',)
        e = exc.JobError(*exc_args, jobid=1234)
        assert 1234 == e.jobid
        with pytest.raises(exc.JobError, match=r'\[jobid=1234\] some error'):
            raise_exc(e)
        assert exc_args == e.args

    def test_reraise_job_error(self):
        try:
            try:
                raise ValueError('random value error')
            except ValueError as e:
                raise exc.JobError('some error', jobid=1234) from e
        except exc.JobError as e:
            assert '[jobid=1234] some error: random value error' == str(e)

    def test_reraise_job_error_no_message(self):
        try:
            try:
                raise ValueError('random value error')
            except ValueError as e:
                raise exc.JobError(jobid=1234) from e
        except exc.JobError as e:
            assert '[jobid=1234]: random value error' == str(e)
