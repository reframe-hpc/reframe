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
        report = line_width * '=' + '\n'
        report += 'SUMMARY OF FAILURES\n'
        for partname, tclist in self._test_cases_bypart.items():
            for tf in [t for t in tclist if t.failed()]:
                check = tf.executor.check
                environ_name = (check.current_environ.name
                                if check.current_environ else 'None')
                report += line_width * '-' + '\n'
                report += 'FAILURE INFO for %s\n' % check.name
                report += '  * System partition: %s\n' % partname
                report += '  * Environment: %s\n' % environ_name
                report += '  * Stage directory: %s\n' % check.stagedir

                job_type = 'local' if check.is_local() else 'batch job'
                jobid = check.job.jobid if check.job else -1
                report += '  * Job type: %s (id=%s)\n' % (job_type, jobid)
                report += '  * Maintainers: %s\n' % check.maintainers
                report += '  * Failing phase: %s\n' % tf.failed_stage
                report += '  * Reason: '
                if tf.exc_info:
                    etype, value, stacktrace = tf.exc_info
                    if isinstance(value, SanityError):
                        report += 'sanity error: %s\n' % value
                    elif isinstance(value, ReframeError):
                        report += 'caught framework exception: %s\n' % value
                    elif isinstance(value, KeyboardInterrupt):
                        report += 'cancelled by user\n'
                    else:
                        report += ('caught unexpected exception: %s (%s)\n' %
                                   (etype.__name__, value))
                        report += ''.join(
                            traceback.format_exception(*tf.exc_info))
                elif tf.failed_stage == 'sanity':
                    report += ('Sanity check failure\n' +
                               check.sanity_info.failure_report())
                elif tf.failed_stage == 'performance':
                    report += ('Performance check failure\n' +
                               check.perf_info.failure_report())
                else:
                    # This shouldn't happen...
                    report += 'Unknown error.'

        report += line_width * '-' + '\n'
        return report
