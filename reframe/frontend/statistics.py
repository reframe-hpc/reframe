import reframe.core.debug as debug
import reframe.core.runtime as rt
from reframe.core.exceptions import StatisticsError


class TestStats:
    '''Stores test case statistics.'''

    def __init__(self):
        # Tasks per run stored as follows: [[run0_tasks], [run1_tasks], ...]
        self._tasks = [[]]

    def __repr__(self):
        return debug.repr(self)

    def add_task(self, task):
        current_run = rt.runtime().current_run
        if current_run == len(self._tasks):
            self._tasks.append([])

        self._tasks[current_run].append(task)

    def tasks(self, run=-1):
        try:
            return self._tasks[run]
        except IndexError:
            raise StatisticsError('no such run: %s' % run) from None

    def failures(self, run=-1):
        return [t for t in self.tasks(run) if t.failed]

    def num_cases(self, run=-1):
        return len(self.tasks(run))

    def retry_report(self):
        # Return an empty report if no retries were done.
        if not rt.runtime().current_run:
            return ''

        line_width = 78
        report = [line_width * '=']
        report.append('SUMMARY OF RETRIES')
        report.append(line_width * '-')
        messages = {}

        for run in range(1, len(self._tasks)):
            for t in self.tasks(run):
                partition_name = ''
                environ_name = ''
                if t.check.current_partition:
                    partition_name = t.check.current_partition.fullname

                if t.check.current_environ:
                    environ_name = t.check.current_environ.name

                key = '%s:%s:%s' % (t.check.name, partition_name, environ_name)
                # Overwrite entry from previous run if available
                messages[key] = (
                    '  * Test %s was retried %s time(s) and %s.' %
                    (t.check.info(), run, 'failed' if t.failed else 'passed')
                )

        for key in sorted(messages.keys()):
            report.append(messages[key])

        return '\n'.join(report)

    def failure_report(self, rerun_failed=-1):
        line_width = 78
        report = [line_width * '=']
        report.append('SUMMARY OF FAILURES')
        current_run = rt.runtime().current_run
        failures = {} 
        for tf in (t for t in self.tasks(current_run) if t.failed):
            check = tf.check
            partition = check.current_partition
            partname = partition.fullname if partition else 'None'
            environ_name = (check.current_environ.name
                            if check.current_environ else 'None')
            retry_info = ('(for the last of %s retries)' % current_run
                          if current_run > 0 else '')

            report.append(line_width * '-')
            report.append('FAILURE INFO for %s %s' % (check.name, retry_info))
            report.append('  * System partition: %s' % partname)
            report.append('  * Environment: %s' % environ_name)
            report.append('  * Stage directory: %s' % check.stagedir)
            report.append('  * Node list: %s' %
                          (','.join(check.job.nodelist)
                           if check.job and check.job.nodelist else '<None>'))
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

            # Collect failures for summary table
            if not tf.failed_stage in failures:
                failures[tf.failed_stage] = []
            failures[tf.failed_stage].append([check.name, environ_name, partname, reason])

        report.append(line_width * '-')

        # Generate summary table
        row_format = "{:<11} {:<5} {:<60}"
        hline = row_format.format(11*'-', 5*'-', 60*'-')
        header = row_format.format('Phase', '#', 'Description')
        stage_descr = {
            'setup': "Failed to set up test's environment and path",
            'compile': "Failed to compile the source code in the current environemnt", 
            'run': "Failed to launch jobs",
            'sanity': "Failed in sanity checking",
            'performance': "Failed in performance checking",
            'poll': "Failed in polling",
            'cleanup': "Failed to clean up the resources of the test." 
        }
        total_num_tests = len(self.tasks(current_run))
        total_num_failures = 0
        for p in failures.keys():
            total_num_failures += len(failures[p])
        summary = ['']
        summary.append('Total number of tests: %d' % int(total_num_tests))
        summary.append('Total number of failures: %d' % int(total_num_failures))
        summary.append('')
        summary.append(header)
        summary.append(hline)
        for p in failures:
            summary.append(row_format.format(p, len(failures[p]), stage_descr[p]))
            if rerun_failed == 1:
                for f in failures[p]:
                    phase = p
                    if "could not load module" in f[3]:
                        phase = "load module"
                    elif "could not show module" in f[3]:
                        phase = "no module"
                    summary.append(row_format.format(
                        '', '', '%s: -n %s -p %s --system %s' % tuple([phase] + f[:3])))
        summary.append('')

        return '\n'.join(report + summary)

    def performance_report(self):
        line_width = 78
        report_start = line_width * '='
        report_title = 'PERFORMANCE REPORT'
        report_end = line_width * '_'
        report_body = []
        previous_name = ''
        previous_part = ''
        for t in self.tasks():
            if t.check.perfvalues.keys():
                if t.check.name != previous_name:
                    report_body.append(line_width * '-')
                    report_body.append('%s' % t.check.name)
                    previous_name = t.check.name

                if t.check.current_partition.fullname != previous_part:
                    report_body.append(
                        '- %s' % t.check.current_partition.fullname)
                    previous_part = t.check.current_partition.fullname

                report_body.append('   - %s' % t.check.current_environ)
                report_body.append('      * num_tasks: %s' % t.check.num_tasks)

            for key, ref in t.check.perfvalues.items():
                var = key.split(':')[-1]
                val = ref[0]
                try:
                    unit = ref[4]
                except IndexError:
                    unit = '(no unit specified)'

                report_body.append('      * %s: %s %s' % (var, val, unit))

        if report_body:
            return '\n'.join([report_start, report_title, *report_body,
                              report_end])

        return ''
