# Copyright 2016-2020 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import json
import jsonschema
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

    def output_dict(self):
        report = []
        current_run = rt.runtime().current_run
        for t in self.tasks(current_run):
            check = t.check
            partition = check.current_partition
            partname = partition.fullname if partition else 'None'
            environ_name = (check.current_environ.name
                            if check.current_environ else 'None')
            report_dict = {
                'name': check.name,
                'description': check.descr,
                'system': partname,
                'environment': environ_name,
                'tags': list(check.tags),
                'maintainers': check.maintainers,
                'scheduler': 'None',
                'jobid': -1,
                'nodelist': [],
                'job_stdout': 'None',
                'job_stderr': 'None'
            }
            if check.job:
                report_dict['scheduler'] = check.job.scheduler.registered_name
                report_dict['jobid'] = (check.job.jobid if check.job.jobid
                                        else -1)
                report_dict['nodelist'] = (check.job.nodelist
                                           if check.job.nodelist
                                           else [])
                report_dict['job_stdout'] = check.job.stdout
                report_dict['job_stderr'] = check.job.stderr

            if check._build_job:
                report_dict['build_stdout'] = check._build_job.stdout
                report_dict['build_stderr'] = check._build_job.stderr

            if t.failed:
                report_dict['result'] = 'fail'
                if t.exc_info is not None:
                    from reframe.core.exceptions import format_exception

                    report_dict['failing_reason'] = format_exception(
                        *t.exc_info)
                    report_dict['failing_phase'] = t.failed_stage
                    report_dict['stagedir'] = (check.stagedir if check.stagedir
                                               else 'None')
            else:
                report_dict['result'] = 'success'
                report_dict['outputdir'] = check.outputdir

            if current_run > 0:
                report_dict['retries'] = current_run

            report.append(report_dict)

        return report

    def json_report(self):
        schema = {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "description": {"type": "string"},
                    "system": {"type": "string"},
                    "environment": {"type": "string"},
                    "stagedir": {"type": "string"},
                    "outputdir": {"type": "string"},
                    "nodelist": {
                        "type": "array",
                        "items": {"type": "string"}
                    },
                    "scheduler": {"type": "string"},
                    "jobid": {"type": "number"},
                    "result": {"type": "string"},
                    "failing_phase": {"type": "string"},
                    "failing_reason": {"type": "string"},
                    "build_stdout": {"type": "string"},
                    "build_stderr": {"type": "string"},
                    "job_stdout": {"type": "string"},
                    "job_stderr": {"type": "string"},
                    "maintainers": {
                        "type": "array",
                        "items": {"type": "string"}
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"}
                    },
                    "retries": {"type": "number"}
                }
            },
        }
        report = self.output_dict()
        jsonschema.validate(instance=report, schema=schema)
        with open('report.json', 'w') as fp:
            json.dump(report, fp, indent=4)

    def failure_report(self):
        line_width = 78
        report = [line_width * '=']
        report.append('SUMMARY OF FAILURES')
        current_run = rt.runtime().current_run
        for tf in (t for t in self.output_dict() if t['result'] == 'fail'):
            retry_info = ('(for the last of %s retries)' % current_run
                          if 'retries' in tf.keys() else '')

            report.append(line_width * '-')
            report.append('FAILURE INFO for %s %s' % (tf['name'], retry_info))
            report.append('  * Test Description: %s' % tf['description'])
            report.append('  * System partition: %s' % tf['system'])
            report.append('  * Environment: %s' % tf['environment'])
            report.append('  * Stage directory: %s' % tf['stagedir'])
            report.append('  * Node list: %s' %
                          (','.join(tf['nodelist'])
                           if tf['nodelist'] else 'None'))
            job_type = 'local' if tf['scheduler'] == 'local' else 'batch job'
            jobid = tf['jobid'] if tf['jobid'] > 0 else 'None'
            report.append('  * Job type: %s (id=%s)' % (job_type, jobid))
            report.append('  * Maintainers: %s' % tf['maintainers'])
            report.append('  * Failing phase: %s' % tf['failing_phase'])
            report.append("  * Rerun with '-n %s -p %s --system %s'" %
                          (tf['name'], tf['environment'], tf['system']))
            report.append("  * Reason: %s" % tf['failing_reason'])

            if tf['failing_phase'] == 'sanity':
                report.append('Sanity check failure')
            elif tf['failing_phase'] == 'performance':
                report.append('Performance check failure')
            else:
                # This shouldn't happen...
                report.append('Unknown error.')

        report.append(line_width * '-')
        return '\n'.join(report)

    def failure_stats(self):
        failures = {}
        current_run = rt.runtime().current_run
        for tf in (t for t in self.tasks(current_run) if t.failed):
            check = tf.check
            partition = check.current_partition
            partname = partition.fullname if partition else 'None'
            environ_name = (check.current_environ.name
                            if check.current_environ else 'None')
            f = f'[{check.name}, {environ_name}, {partname}]'
            if tf.failed_stage not in failures:
                failures[tf.failed_stage] = []

            failures[tf.failed_stage].append(f)

        line_width = 78
        stats_start = line_width * '='
        stats_title = 'FAILURE STATISTICS'
        stats_end = line_width * '-'
        stats_body = []
        row_format = "{:<13} {:<5} {}"
        stats_hline = row_format.format(13*'-', 5*'-', 60*'-')
        stats_header = row_format.format('Phase', '#', 'Failing test cases')
        num_tests = len(self.tasks(current_run))
        num_failures = 0
        for l in failures.values():
            num_failures += len(l)

        stats_body = ['']
        stats_body.append('Total number of test cases: %s' % num_tests)
        stats_body.append('Total number of failures: %s' % num_failures)
        stats_body.append('')
        stats_body.append(stats_header)
        stats_body.append(stats_hline)
        for p, l in failures.items():
            stats_body.append(row_format.format(p, len(l), l[0]))
            for f in l[1:]:
                stats_body.append(row_format.format('', '', str(f)))

        if stats_body:
            return '\n'.join([stats_start, stats_title, *stats_body,
                              stats_end])
        return ''

    def performance_report(self):
        line_width = 78
        report_start = line_width * '='
        report_title = 'PERFORMANCE REPORT'
        report_end = line_width * '-'
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
