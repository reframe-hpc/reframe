import traceback
import reframe.core.debug as debug

from reframe.core.exceptions import ReframeError, SanityError


class TestStats:
    """Stores test case statistics."""

    def __init__(self, test_cases=None):
        test_cases = test_cases or []
        if not hasattr(test_cases, '__iter__'):
            raise TypeError('expected an iterable')

        # Store test cases per partition internally
        self._test_cases_bypart = {}
        for t in test_cases:
            partition = t.executor.check.current_partition
            partname = partition.fullname if partition else 'None'

            tclist = self._test_cases_bypart.setdefault(partname, [])
            tclist.append(t)

    def __repr__(self):
        return debug.repr(self)

    def num_failures(self, partition=None):
        num_fails = 0
        if partition:
            num_fails += len([
                t for t in self._test_cases_bypart[partition] if t.failed()
            ])
        else:
            # count all failures
            for tclist in self._test_cases_bypart.values():
                num_fails += len([t for t in tclist if t.failed()])

        return num_fails

    def num_failures_stage(self, stage):
        num_fails = 0
        for tclist in self._test_cases_bypart.values():
            num_fails += len([t for t in tclist if t.failed_stage == stage])

        return num_fails

    def num_cases(self, partition=None):
        num_cases = 0
        if partition:
            num_cases += len(self._test_cases_bypart[partition])
        else:
            # count all failures
            for tclist in self._test_cases_bypart.values():
                num_cases += len(tclist)

        return num_cases

    def failure_report(self):
        line_width = 78
        report = [line_width * '=']
        report.append('SUMMARY OF FAILURES')
        for partname, tclist in self._test_cases_bypart.items():
            for tf in (t for t in tclist if t.failed()):
                check = tf.executor.check
                environ_name = (check.current_environ.name
                                if check.current_environ else 'None')
                report.append(line_width * '-')
                report.append('FAILURE INFO for %s' % check.name)
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

                elif tf.failed_stage == 'sanity':
                    report.append('Sanity check failure')
                elif tf.failed_stage == 'performance':
                    report.append('Performance check failure')
                else:
                    # This shouldn't happen...
                    report.append('Unknown error.')

        report.append(line_width * '-')
        return '\n'.join(report)
