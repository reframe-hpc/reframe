import json
import os
import tempfile

import reframe as rfm
import reframe.utility.osext as osext
import reframe.utility.systeminfo as sysinfo
from reframe.core.logging import getlogger
from reframe.core.runtime import runtime
from reframe.core.schedulers import Job


def _load_topology(filename):
    try:
        with open(filename) as fp:
            return json.load(fp)
    except OSError as e:
        getlogger().warning(
            f'could not load topology file: {filename!r}: {e}'
        )
        return {}


def _save_topology(filename, topo_info):
    if not topo_info:
        return

    os.makedirs(os.path.dirname(filename), exist_ok=True)
    try:
        with open(filename, 'w') as fp:
            json.dump(topo_info, fp, indent=2)
    except OSError as e:
        getlogger().warning(
            f'could not save topology file: {filename!r}: {e}'
        )


def _is_part_local(part):
    return (part.scheduler.registered_name == 'local' and
            part.launcher_type.registered_name == 'local')


def _remote_detect(part):
    getlogger().info(
        f'Detecting topology of remote partition {part.fullname!r}'
    )
    rfm_exec = os.path.join(rfm.INSTALL_PREFIX, 'bin/reframe')
    try:
        with tempfile.TemporaryDirectory(dir='.') as dirname:
            job = Job.create(part.scheduler,
                             part.launcher_type(),
                             name='rfm-detect-job',
                             sched_access=part.access)
            with osext.change_dir(dirname):
                launcher_cmd = job.launcher.run_command(job)
                job.prepare([f'{launcher_cmd} {rfm_exec} '
                             f'--detect-host-topology=topo.json'],
                            trap_errors=True)
                with open(job.script_filename) as fp:
                    getlogger().debug(
                        f'submitting remote job script:\n{fp.read()}'
                    )

                job.submit()
                job.wait()
                with open('topo.json') as fp:
                    topo_info = json.load(fp)

    except Exception as e:
        getlogger().warning(f'failed to retrieve remote processor info: {e}')
        topo_info = {}

    return topo_info


def detect_topology():
    rt = runtime()
    detect_remote_systems = rt.get_option(
        'general/0/detect_remote_system_topology'
    )
    config_file = rt.site_config.filename
    if config_file == '<builtin>':
        config_prefix = os.path.join(
            os.getenv('HOME'), '.reframe/topology'
        )
    else:
        config_prefix = os.path.dirname(config_file)
        config_prefix = os.path.join(config_prefix, '_meta')

    for part in rt.system.partitions:
        getlogger().debug(f'detecting topology info for {part.fullname}')
        if part.processor.info != {}:
            # Processor info set up already in the configuration
            getlogger().debug(
                f'> topology found in configuration file; skipping...'
            )
            continue

        topo_file = os.path.join(
            config_prefix, f'{rt.system.name}-{part.name}', 'processor.json'
        )
        if os.path.exists(topo_file):
            getlogger().debug(
                f'> found topology file {topo_file!r}; loading...'
            )
            part.processor._info = _load_topology(topo_file)
            continue

        # No topology found, try to auto-detect it
        getlogger().debug(f'> no topology file found; auto-detecting...')
        if _is_part_local(part):
            # Unconditionally detect the system for fully local partitions
            part.processor._info = sysinfo.get_proc_info()
            _save_topology(topo_file, part.processor.info)
        elif detect_remote_systems:
            part.processor._info = _remote_detect(part)
            _save_topology(topo_file, part.processor.info)

        getlogger().debug(f'> saved topology in {topo_file!r}')
