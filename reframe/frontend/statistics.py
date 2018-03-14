import reframe.core.debug as debug


class TestStats:
    """Stores test case statistics."""

    def __init__(self, tasks=[]):
        if not hasattr(tasks, '__iter__'):
            raise TypeError('expected an iterable')

        self._tasks = tasks
        self._last_retry = max([t.retry_num for t in tasks])

    def __repr__(self):
        return debug.repr(self)

    # The retry_num can be indexed from the end (e.g. -1 for the last retry)
    def _convert_retry_num(self, retry_num):
        return self._last_retry + retry_num + 1 if retry_num < 0 else retry_num

    def last_retry(self):
        return self._last_retry

    def num_success_all_retries(self):
        retry_nums = range(1, self._last_retry)
        return len([t for t in self._tasks if not t.failed and t.retry_num in
                                                               retry_nums])

    def num_success(self, retry_num=0):
        retry_num = self._convert_retry_num(retry_num)
        return len([t for t in self._tasks if not t.failed and t.retry_num ==
                                                               retry_num])

    def num_failures(self, retry_num=0):
        retry_num = self._convert_retry_num(retry_num)
        return len([t for t in self._tasks if t.failed and t.retry_num ==
                                                           retry_num])

    def num_failures_stage(self, stage, retry_num=0):
        retry_num = self._convert_retry_num(retry_num)
        return len([t for t in self._tasks if t.failed_stage == stage and
                                              t.retry_num == retry_num])

    def num_cases(self, retry_num=0):
        retry_num = self._convert_retry_num(retry_num)
        return len([t for t in self._tasks if t.retry_num == retry_num])

    def check_names_failed(self, retry_num=0):
        retry_num = self._convert_retry_num(retry_num)
        return set([t.check.name for t in self._tasks
                    if t.failed and t.retry_num == retry_num])

    def failure_report(self, retry_num=0):
        retry_num = self._convert_retry_num(retry_num)
        line_width = 78
        report = [line_width * '=']
        report.append('SUMMARY OF FAILURES')
        for tf in (t for t in self._tasks if t.failed and
                                             t.retry_num == retry_num):
            check = tf.check
            partition = check.current_partition
            partname = partition.fullname if partition else 'None'
            environ_name = (check.current_environ.name
                            if check.current_environ else 'None')
            retry_info = ('(for the last of %s retries)' % tf.retry_num
                          if tf.retry_num > 0 else '')

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
