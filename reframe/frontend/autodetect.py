import json
import os
import tempfile

import reframe as rfm
import reframe.core.shell as shell
import reframe.utility.osext as osext
import reframe.utility.systeminfo as sysinfo
from reframe.core.logging import getlogger
from reframe.core.runtime import runtime
from reframe.core.schedulers import Job


# reframe --detect-local-topology
#
# RFM_DETECT_REMOTE_SYSTEM_TOPOLOGY=y (default=n)
#
# ReFrame will launch remote jobs executing `reframe --detect-local-topology`

def _load_procinfo(filename):
    try:
        with open(filename) as fp:
            return json.load(fp)
    except OSError as e:
        getlogger().warning(
            f'could not load procinfo file: {filename!r}: {e}'
        )
        return {}


def _save_procinfo(filename, procinfo):
    if not procinfo:
        return

    os.makedirs(os.path.dirname(filename), exist_ok=True)
    try:
        with open(filename, 'w') as fp:
            json.dump(procinfo, fp, indent=2)
    except OSError as e:
        getlogger().warning(
            f'could not save procinfo file: {filename!r}: {e}'
        )


def _is_part_local(part):
    return (part.scheduler.registered_name == 'local' and
            part.launcher_type.registered_name == 'local')


def _remote_detect(part):
    rfm_exec = os.path.join(rfm.INSTALL_PREFIX, 'bin/reframe')
    try:
        with tempfile.TemporaryDirectory(dir='.') as dirname:
            job = Job.create(part.scheduler,
                             part.launcher_type(),
                             name='rfm-detect-job',
                             sched_access=part.access)
            with osext.change_dir(dirname):
                job.prepare([f'{rfm_exec} --detect-local-topology=topo.json'],
                            trap_errors=True)
                job.submit()
                job.wait()
                with open('topo.json') as fp:
                    procinfo = json.load(fp)
    except Exception as e:
        getlogger().warning(f'failed to retrieve remote processor info: {e}')
        procinfo = {}

    return procinfo


def detect_procinfo():
    rt = runtime()
    detect_remote_systems = rt.get_option(
        'general/0/detect_remote_system_topology'
    )
    config_file = rt.site_config.filename
    if config_file == '<builtin>':
        config_prefix = os.path.join(
            os.getenv('HOME'), '.reframe/procinfo'
        )
    else:
        config_prefix = os.path.dirname(config_file)
        config_prefix = os.path.join(config_prefix, '_meta')

    for part in rt.system.partitions:
        if part.processor.info != {}:
            # Processor info set up already in the configuration
            continue

        procinfo_file = os.path.join(
            config_prefix, f'{rt.system.name}-{part.name}', 'processor.json'
        )

        if os.path.exists(procinfo_file):
            part.processor._info = _load_procinfo(procinfo_file)
            continue

        # No procinfo found, try to auto-detect it
        if _is_part_local(part):
            # Unconditionally detect the system for fully local partitions
            part.processor._info = sysinfo.get_proc_info()
            _save_procinfo(procinfo_file, part.processor.info)
        elif detect_remote_systems:
            part.processor._info = _remote_detect(part)
            _save_procinfo(procinfo_file, part.processor.info)
