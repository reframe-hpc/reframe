import reframe.core.debug as debug
from reframe.core.exceptions import (ReframeFatalError, StatisticsError)


class TestStats:
    """Stores test case statistics."""

    def __init__(self):
        self._tasks = [[]]
        self._current_run = 0

    def __repr__(self):
        return debug.repr(self)

    def next_run(self, current_run):
        if current_run != self._current_run + 1:
            raise ReframeFatalError('current_run variable out of sync')

        self._current_run = current_run
        self._tasks.append([])

    def add_task(self, task):
        self._tasks[self._current_run].append(task)

    def _get_tasks(self, run):
        try:
            return self._tasks[run]
        except IndexError:
            raise StatisticsError('no such run: %s' % run)

    def num_failures(self, run=-1):
        return len([t for t in self._get_tasks(run) if t.failed])

    def num_cases(self, run=-1):
        return len(self._get_tasks(run))

    def tasks_failed(self, run=-1):
        return [t for t in self._get_tasks(run) if t.failed]

    def retry_report(self):
        # Return an empty report if no retries were done.
        if not self._current_run:
            return ''

        line_width = 78
        report = [line_width * '=']
        report.append('SUMMARY OF RETRIES')
        report.append(line_width * '-')
        messages = {}
        for run in range(1, len(self._tasks)):
            for t in self._get_tasks(run):
                # Overwrite entry from previous run if available
                messages[t.check.info()] = (
                    '  * Test %s was retried %s time(s) and %s.' %
                    (t.check.info(), run, 'failed' if t.failed else 'passed')
                )

        for key in sorted(messages.keys()):
            report.append(messages[key])

        return '\n'.join(report)

    def failure_report(self):
        line_width = 78
        report = [line_width * '=']
        report.append('SUMMARY OF FAILURES')
        for tf in (t for t in self._get_tasks(self._current_run) if t.failed):
            check = tf.check
            partition = check.current_partition
            partname = partition.fullname if partition else 'None'
            environ_name = (check.current_environ.name
                            if check.current_environ else 'None')
            retry_info = ('(for the last of %s retries)' % self._current_run
                          if self._current_run > 0 else '')

            report.append(line_width * '-')
            report.append('FAILURE INFO for %s %s' % (check.name, retry_info))
            report.append('  * System partition: %s' % partname)
            report.append('  * Environment: %s' % environ_name)
            report.append('  * Stage directory: %s' % check.stagedir)

            job_type = 'local' if check.is_local() else 'batch job'
            jobid = check.job.jobid if check.job else -1
            report.append('  * Job type: %s (id=%s)' % (job_type, jobid))
            report.append('  * Maintainers: %s' % check.maintainers)
            report.append('  * Failing phase: %s' % tf.failed_stage)
            reason = '  * Reason: '
            if tf.exc_info is not None:
                from reframe.core.exceptions import format_exception

                reason += format_exception(*tf.exc_info)
                report.append(reason)

            elif tf.failed_stage == 'check_sanity':
                report.append('Sanity check failure')
            elif tf.failed_stage == 'check_performance':
                report.append('Performance check failure')
            else:
                # This shouldn't happen...
                report.append('Unknown error.')

        report.append(line_width * '-')
        return '\n'.join(report)
