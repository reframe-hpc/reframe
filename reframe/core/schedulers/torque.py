import re

from reframe.core.config import settings
from reframe.core.exceptions import JobError
from reframe.core.logging import getlogger
from reframe.core.schedulers.pbs import PbsJobScheduler, _run_strict
from reframe.core.schedulers.registry import register_scheduler
import reframe.utility.os_ext as os_ext


JOB_STATES = {
    'Q': 'QUEUED',
    'H': 'HELD',
    'R': 'RUNNING',
    'E': 'EXITING',
    'T': 'MOVED',
    'W': 'WAITING',
    'S': 'SUSPENDED',
    'C': 'COMPLETED',
}


@register_scheduler('torque')
class TorqueJobScheduler(PbsJobScheduler):
    def _update_state(self, job):
        '''Check the status of the job.'''

        completed = os_ext.run_command('qstat -f %s' % job.jobid, log=False)

        # Needed for clusters that have the keep_completed parameter set to 0
        if completed.returncode == 153:
            getlogger().debug(
                'jobid not known by scheduler, assuming job completed'
            )
            job.state = 'COMPLETED'
            return

        if completed.returncode != 0:
            raise JobError(completed.returncode, completed.stderr)

        state_match = re.search(
            r'^\s*job_state = (?P<state>[A-Z])', completed.stdout, re.MULTILINE
        )
        if not state_match:
            getlogger().debug(
                'job state not found (stdout follows)\n%s' % completed.stdout
            )
            return

        state = state_match.group('state')
        job.state = JOB_STATES[state]

        if job.state == 'COMPLETED':
            code_match = re.search(
                r'^\s*exit_status = (?P<code>\d+)',
                completed.stdout,
                re.MULTILINE,
            )
            if not code_match:
                return
            job.exitcode = int(code_match.group('code'))

    def _emit_lselect_option(self, job):
        num_tasks_per_node = job.num_tasks_per_node or 1
        num_cpus_per_task = job.num_cpus_per_task or 1
        num_nodes = job.num_tasks // num_tasks_per_node
        num_cpus_per_node = num_tasks_per_node * num_cpus_per_task

        select_opt = '-l nodes=%s:ppn=%s' % (num_nodes, num_cpus_per_node)

        # Options starting with `-` are emitted in separate lines
        rem_opts = []
        verb_opts = []
        for opt in (*job.sched_access, *job.options):
            if opt.startswith('-'):
                rem_opts.append(opt)
            elif opt.startswith('#'):
                verb_opts.append(opt)
            else:
                select_opt += ':' + opt

        return [
            self._format_option(select_opt),
            *(self._format_option(opt) for opt in rem_opts),
            *verb_opts,
        ]

    def finished(self, job):
        try:
            self._update_state(job)
        except JobError as e:
            # We ignore these exceptions at this point and we simply mark the
            # job as unfinished.
            getlogger().debug('ignoring error during polling: %s' % e)
            return False
        else:
            return job.state == 'COMPLETED'

    def submit(self, job):
        # `-o` and `-e` options are only recognized in command line by the PBS
        # Slurm wrappers.
        cmd = 'qsub -o %s -e %s %s' % (
            job.stdout,
            job.stderr,
            job.script_filename,
        )
        completed = _run_strict(cmd, timeout=settings().job_submit_timeout)
        jobid_match = re.search(r'^(?P<jobid>\S+)', completed.stdout)
        if not jobid_match:
            raise JobError(
                'could not retrieve the job id ' 'of the submitted job'
            )

        jobid, *info = jobid_match.group('jobid').split('.', maxsplit=1)
        job.jobid = int(jobid)
        if info:
            self._pbs_server = info[0]
