# Copyright 2016-2021 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import json
import os
import tempfile

import reframe as rfm
import reframe.utility.osext as osext
from reframe.core.logging import getlogger
from reframe.core.runtime import runtime
from reframe.core.schedulers import Job
from reframe.utility.cpuinfo import cpuinfo


def _load_info(filename):
    try:
        with open(filename) as fp:
            return json.load(fp)
    except OSError as e:
        getlogger().warning(
            f'could not load file: {filename!r}: {e}'
        )
        return {}


def _save_info(filename, topo_info):
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
        found_procinfo = False
        found_devinfo  = False
        if part.processor.info != {}:
            # Processor info set up already in the configuration
            getlogger().debug(
                f'> topology found in configuration file; skipping...'
            )
            found_procinfo = True

        if part.devices:
            # Devices set up already in the configuration
            getlogger().debug(
                f'> devices found in configuration file; skipping...'
            )
            found_devinfo = True

        if found_procinfo and found_devinfo:
            continue

        topo_file = os.path.join(
            config_prefix, f'{rt.system.name}-{part.name}', 'processor.json'
        )
        dev_file = os.path.join(
            config_prefix, f'{rt.system.name}-{part.name}', 'devices.json'
        )
        if not found_procinfo and os.path.exists(topo_file):
            getlogger().debug(
                f'> found topology file {topo_file!r}; loading...'
            )
            part.processor._info = _load_info(topo_file)
            found_procinfo = True

        if not found_devinfo and os.path.exists(dev_file):
            getlogger().debug(
                f'> found devices file {dev_file!r}; loading...'
            )
            part._devices = _load_info(dev_file)
            found_devinfo = True

        if found_procinfo and found_devinfo:
            continue

        if not found_procinfo:
            # No topology found, try to auto-detect it
            getlogger().debug(f'> no topology file found; auto-detecting...')
            if _is_part_local(part):
                # Unconditionally detect the system for fully local partitions
                part.processor._info = cpuinfo()
                _save_info(topo_file, part.processor.info)
            elif detect_remote_systems:
                part.processor._info = _remote_detect(part)
                _save_info(topo_file, part.processor.info)

            getlogger().debug(f'> saved topology in {topo_file!r}')

        if not found_devinfo:
            getlogger().debug(f'> device auto-detection is not supported')
