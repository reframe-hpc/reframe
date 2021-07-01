# Copyright 2016-2021 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import json
import jsonschema
import os
import tempfile

import reframe as rfm
import reframe.utility.osext as osext
from reframe.core.exceptions import ConfigError
from reframe.core.logging import getlogger
from reframe.core.runtime import runtime
from reframe.core.schedulers import Job
from reframe.utility.cpuinfo import cpuinfo


_REFRAME_GH_REPO = 'https://github.com/eth-cscs/reframe.git'


def _subschema(fragment):
    '''Create a configuration subschema.'''

    full_schema = runtime().site_config.schema
    return {
        '$schema': full_schema['$schema'],
        'defs': full_schema['defs'],
        '$ref': fragment
    }


def _validate_info(info, schema):
    if schema is None:
        return info

    jsonschema.validate(info, schema)
    return info


def _load_info(filename, schema=None):
    try:
        with open(filename) as fp:
            return _validate_info(json.load(fp), schema)
    except OSError as e:
        getlogger().warning(
            f'could not load file: {filename!r}: {e}'
        )
        return {}
    except jsonschema.ValidationError as e:
        raise ConfigError(
            f'could not validate meta-config file {filename!r}'
        ) from e


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
    def _emit_script(job, fresh=False):
        if fresh:
            commands = [
                f'_prefix=$(mktemp -d)',
                f'cd $_prefix',
                f'git clone {_REFRAME_GH_REPO} reframe',
                f'cd reframe',
                f'./bootstrap.sh'
            ]
        else:
            commands = []

        launcher_cmd = job.launcher.run_command(job)
        commands += [
            f'{launcher_cmd} {rfm_exec} --detect-host-topology'
        ]
        job.prepare(commands, trap_errors=True)

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
                more_tries = [{'fresh': False}]
                while more_tries:
                    args = more_tries.pop()

                    _emit_script(job, **args)
                    with open(job.script_filename) as fp:
                        getlogger().debug(
                            f'submitting remote job script:\n{fp.read()}'
                        )

                    job.submit()
                    job.wait()
                    with open(job.stdout) as fp:
                        try:
                            topo_info = json.load(fp)
                        except json.JSONDecodeError:
                            getlogger().debug(f'not a JSON file:\n{fp.read()}')
                            more_tries.append({'fresh': True})
    except Exception as e:
        topo_info = {}

    if not topo_info:
        getlogger().warning(f'failed to retrieve remote processor info: {e}')

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
        config_prefix = os.path.join(os.path.dirname(config_file), '_meta')

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
            part.processor._info = _load_info(topo_file,
                                              _subschema('#/defs/processor_info'))
            found_procinfo = True

        if not found_devinfo and os.path.exists(dev_file):
            getlogger().debug(
                f'> found devices file {dev_file!r}; loading...'
            )
            part._devices = _load_info(dev_file, _subschema('#/defs/devices'))
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
