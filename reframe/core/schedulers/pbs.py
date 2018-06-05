import itertools
import os
import re
import time
from datetime import datetime

import reframe.core.schedulers as sched
import reframe.utility.os_ext as os_ext
from reframe.core.exceptions import (SpawnedProcessError,
                                     JobBlockedError, JobError)
from reframe.core.logging import getlogger
from reframe.core.schedulers.registry import register_scheduler
from reframe.settings import settings


@register_scheduler('pbs')
class PbsJob(sched.Job):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._prefix  = '#PBS'
        self._is_cancelling = False
        # fix for regression tests with compile
        if os.path.dirname(self.command) is '.':
            self._command = os.path.join(
                self.workdir, os.path.basename(self.command))

    def _emit_job_option(self, var, option, builder):
        if var is not None:
            if isinstance(var, tuple):
                builder.verbatim(self._prefix + ' ' + option.format(*var))
            else:
                builder.verbatim(self._prefix + ' ' + option.format(var))

    def _run_command(self, cmd, timeout=None):
        """Run command cmd and re-raise any exception as a JobError."""
        try:
            return os_ext.run_command(cmd, check=True, timeout=timeout)
        except SpawnedProcessError as e:
            raise JobError(jobid=self._jobid) from e

    def emit_preamble(self, builder):
        self._emit_job_option(self.name, '-N "{0}"', builder)

        extra_options = ''
        if len(self.options):
            extra_options = ':' + ':'.join(self.options)

        self._emit_job_option((int(self._num_tasks/self._num_tasks_per_node), self._num_tasks_per_node, self._num_tasks_per_node, extra_options),
                              '-lselect={0}:ncpus={1}:mpiprocs={2}{3}', builder)
        self._emit_job_option('%d:%d:%d' % self.time_limit,
                              '-l walltime={0}', builder)

        self._emit_job_option(self.sched_partition, '-q {0}', builder)

        self._emit_job_option(self.stdout, '-o {0}', builder)
        self._emit_job_option(self.stderr, '-e {0}', builder)

    def submit(self):
        cmd = 'qsub %s' % self.script_filename
        completed = self._run_command(cmd, settings.job_submit_timeout)
        jobid_match = re.search('^(?P<jobid>\d+)',
                                completed.stdout)
        full_jobid_match = re.search('^(?P<fjobid>\d+\.\w+\d*)$',
                                     completed.stdout)

        if not jobid_match:
            raise JobError(
                'could not retrieve the job id of the submitted job')

        self._jobid = int(jobid_match.group('jobid'))
        self._fulljobid = full_jobid_match.group('fjobid')

    def wait(self):
        super().wait()
        intervals = itertools.cycle(settings.job_poll_intervals)

        # check if the stdout file is already there (i.e. job finished)
        while not os.path.isfile(self.stdout):
            time.sleep(next(intervals))

    def cancel(self):
        super().cancel()
        getlogger().debug('cancelling job (id=%s)' % self._jobid)
        self._is_cancelling = True
        jobid, server = self._fulljobid.split(".")
        if server == 'pbspro':
            self._run_command('qdel %s' % self._fulljobid,
                              settings.job_submit_timeout)
        elif server == 'pbs11':
            self._run_command('qdel %s@pbs11' % self._fulljobid,
                              settings.job_submit_timeout)
        else:
            raise JobError('Did not recognize server', server)

    def finished(self):
        super().finished()
        intervals = itertools.cycle(settings.job_poll_intervals)
        if os.path.isfile(self.stdout):
            return True
        else:
            return False
