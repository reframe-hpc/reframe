from itertools import chain

import reframe.core.debug as debug
import reframe.core.exceptions as exc


class TestStats:
    """Stores test case statistics."""

    def __init__(self):
        self._tasks = []
        self._try_num = -1

    def __repr__(self):
        return debug.repr(self)

    def next_try(self, try_num):
        # TODO: RAISE DIFFERENT error?
        if try_num != self._try_num + 1:
            raise exc.ReframeFatalError('try_num out of sync')

        self._try_num = try_num
        self._tasks.append([])

    def add_task(self, task):
        self._tasks[self._try_num].append(task)

    # The try_num can be indexed from the end (e.g. -1 for the last retry)
    def _convert_try_num(self, try_num):
        return self._try_num + try_num + 1 if try_num < 0 else try_num

    def num_success_all_retries(self):
        tasks_retries = list(chain.from_iterable(self._tasks[1:]))
        return len([t for t in tasks_retries if not t.failed])

    def num_success(self, try_num=0):
        try_num = self._convert_try_num(try_num)
        return len([t for t in self._tasks[try_num] if not t.failed])

    def num_failures(self, try_num=0):
        try_num = self._convert_try_num(try_num)
        return len([t for t in self._tasks[try_num] if t.failed])

    def num_failures_stage(self, stage, try_num=0):
        try_num = self._convert_try_num(try_num)
        return len([t for t in self._tasks[try_num] if t.failed_stage ==
                                                         stage])

    def num_cases(self, try_num=0):
        try_num = self._convert_try_num(try_num)
        return len(self._tasks[try_num])

    def check_names_failed(self, try_num=0):
        try_num = self._convert_try_num(try_num)
        return set([t.check.name for t in self._tasks[try_num] if t.failed])

    def failure_report(self, try_num=0):
        try_num = self._convert_try_num(try_num)
        line_width = 78
        report = [line_width * '=']
        report.append('SUMMARY OF FAILURES')
        for tf in (t for t in self._tasks[try_num] if t.failed):
            check = tf.check
            partition = check.current_partition
            partname = partition.fullname if partition else 'None'
            environ_name = (check.current_environ.name
                            if check.current_environ else 'None')
            retry_info = ('(for the last of %s retries)' % try_num
                          if try_num > 0 else '')

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
