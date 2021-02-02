# Copyright 2016-2021 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import json

import reframe.utility as utility
import reframe.utility.jsonext as jsonext
from reframe.core.backends import (getlauncher, getscheduler)
from reframe.core.logging import getlogger
from reframe.core.modules import ModulesSystem
from reframe.core.environments import (Environment, ProgEnvironment)


class SystemPartition(jsonext.JSONSerializable):
    '''A representation of a system partition inside ReFrame.

    .. warning::
       Users may not create :class:`SystemPartition` objects directly.
    '''

    def __init__(self, parent, name, sched_type, launcher_type,
                 descr, access, container_environs, resources,
                 local_env, environs, max_jobs):
        getlogger().debug(f'Initializing system partition {name!r}')
        self._parent_system = parent
        self._name = name
        self._sched_type = sched_type
        self._scheduler = None
        self._launcher_type = launcher_type
        self._descr = descr
        self._access = access
        self._container_environs = container_environs
        self._local_env = local_env
        self._environs = environs
        self._max_jobs = max_jobs
        self._resources = {r['name']: r['options'] for r in resources}

    @property
    def access(self):
        '''The scheduler options for accessing this system partition.

        :type: :class:`List[str]`
        '''
        return utility.SequenceView(self._access)

    @property
    def descr(self):
        '''The description of this partition.

        :type: :class:`str`
        '''
        return self._descr

    @property
    def environs(self):
        '''The programming environments associated with this system partition.

        :type: :class:`List[ProgEnvironment]`
        '''

        return utility.SequenceView(self._environs)

    @property
    def container_environs(self):
        '''Environments associated with the different container platforms.

        :type: :class:`Dict[str, Environment]`
        '''

        return utility.MappingView(self._container_environs)

    @property
    def fullname(self):
        '''Return the fully-qualified name of this partition.

        The fully-qualified name is of the form
        ``<parent-system-name>:<partition-name>``.

        :type: :class:`str`
        '''
        return f'{self._parent_system}:{self._name}'

    @property
    def local_env(self):
        '''The local environment associated with this partition.

        :type: :class:`Environment`
        '''
        return self._local_env

    @property
    def max_jobs(self):
        '''The maximum number of concurrent jobs allowed on this partition.

        :type: integral
        '''
        return self._max_jobs

    @property
    def name(self):
        '''The name of this partition.

        :type: :class:`str`
        '''
        return self._name

    @property
    def resources(self):
        '''The resources template strings associated with this partition.

        This is a dictionary, where the key is the name of a resource and the
        value is the scheduler options or directives associated with this
        resource.

        :type: :class:`Dict[str, List[str]]`

        '''

        return utility.MappingView(self._resources)

    @property
    def scheduler(self):
        '''The backend scheduler of this partition.

        :type: :class:`reframe.core.schedulers.JobScheduler`.

        .. note::
           .. versionchanged:: 2.8
              Prior versions returned a string representing the scheduler and
              job launcher combination.
           .. versionchanged:: 3.2
              The property now stores a :class:`JobScheduler` instance.

        '''
        if self._scheduler is None:
            self._scheduler = self._sched_type()

        return self._scheduler

    @property
    def launcher_type(self):
        '''The type of the backend launcher of this partition.

        .. versionadded:: 3.2

        :type: a subclass of :class:`reframe.core.launchers.JobLauncher`.
        '''
        return self._launcher_type

    @property
    def launcher(self):
        '''See :attr:`launcher_type`.

        .. deprecated:: 3.2
           Please use :attr:`launcher_type` instead.
        '''

        from reframe.core.warnings import user_deprecation_warning

        user_deprecation_warning("the 'launcher' attribute is deprecated; "
                                 "please use 'launcher_type' instead")
        return self.launcher_type

    def get_resource(self, name, **values):
        '''Instantiate managed resource ``name`` with ``value``.

        :meta private:
        '''

        ret = []
        for r in self._resources.get(name, []):
            try:
                ret.append(r.format(**values))
            except KeyError:
                pass

        return ret

    def environment(self, name):
        '''Return the partition environment named ``name``.'''

        for e in self.environs:
            if e.name == name:
                return e

        return None

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented

        return (self._name == other.name and
                self._sched_type == other._sched_type and
                self._launcher_type == other._launcher_type and
                self._access == other._access and
                self._environs  == other._environs and
                self._resources == other._resources and
                self._local_env == other._local_env)

    def __hash__(self):
        return hash(self.fullname)

    def json(self):
        '''Return a JSON object representing this system partition.'''

        return {
            'name': self._name,
            'descr': self._descr,
            'scheduler': self._sched_type.registered_name,
            'launcher': self._launcher_type.registered_name,
            'access': self._access,
            'container_platforms': [
                {
                    'type': ctype,
                    'modules': [m for m in cpenv.modules],
                    'variables': [[n, v] for n, v in cpenv.variables.items()]
                }
                for ctype, cpenv in self._container_environs.items()
            ],
            'modules': [m for m in self._local_env.modules],
            'variables': [[n, v]
                          for n, v in self._local_env.variables.items()],
            'environs': [e.name for e in self._environs],
            'max_jobs': self._max_jobs,
            'resources': [
                {
                    'name': name,
                    'options': options
                }
                for name, options in self._resources.items()
            ]
        }

    def __str__(self):
        return json.dumps(self.json(), indent=2)


class System(jsonext.JSONSerializable):
    '''A representation of a system inside ReFrame.

    .. warning::
       Users may not create :class:`System` objects directly.
    '''

    def __init__(self, name, descr, hostnames, modules_system,
                 preload_env, prefix, outputdir,
                 resourcesdir, stagedir, partitions):
        getlogger().debug(f'Initializing system {name!r}')
        self._name = name
        self._descr = descr
        self._hostnames = hostnames
        self._modules_system = ModulesSystem.create(modules_system)
        self._preload_env = preload_env
        self._prefix = prefix
        self._outputdir = outputdir
        self._resourcesdir = resourcesdir
        self._stagedir = stagedir
        self._partitions = partitions

    @classmethod
    def create(cls, site_config):
        # Create the whole system hierarchy from bottom up
        sysname = site_config.get('systems/0/name')
        partitions = []
        config_save = site_config.subconfig_system
        for p in site_config.get('systems/0/partitions'):
            site_config.select_subconfig(f'{sysname}:{p["name"]}')
            partid = f"systems/0/partitions/@{p['name']}"
            part_name = site_config.get(f'{partid}/name')
            part_sched = getscheduler(site_config.get(f'{partid}/scheduler'))
            part_launcher = getlauncher(site_config.get(f'{partid}/launcher'))
            part_container_environs = {}
            for i, p in enumerate(
                    site_config.get(f'{partid}/container_platforms')
            ):
                ctype = p['type']
                part_container_environs[ctype] = Environment(
                    name=f'__rfm_env_{ctype}',
                    modules=site_config.get(
                        f'{partid}/container_platforms/{i}/modules'
                    ),
                    variables=site_config.get(
                        f'{partid}/container_platforms/{i}/variables'
                    )
                )

            part_environs = [
                ProgEnvironment(
                    name=e,
                    modules=site_config.get(f'environments/@{e}/modules'),
                    variables=site_config.get(f'environments/@{e}/variables'),
                    cc=site_config.get(f'environments/@{e}/cc'),
                    cxx=site_config.get(f'environments/@{e}/cxx'),
                    ftn=site_config.get(f'environments/@{e}/ftn'),
                    cppflags=site_config.get(f'environments/@{e}/cppflags'),
                    cflags=site_config.get(f'environments/@{e}/cflags'),
                    cxxflags=site_config.get(f'environments/@{e}/cxxflags'),
                    fflags=site_config.get(f'environments/@{e}/fflags'),
                    ldflags=site_config.get(f'environments/@{e}/ldflags')
                ) for e in site_config.get(f'{partid}/environs')
            ]
            partitions.append(
                SystemPartition(
                    parent=site_config.get('systems/0/name'),
                    name=part_name,
                    sched_type=part_sched,
                    launcher_type=part_launcher,
                    descr=site_config.get(f'{partid}/descr'),
                    access=site_config.get(f'{partid}/access'),
                    resources=site_config.get(f'{partid}/resources'),
                    environs=part_environs,
                    container_environs=part_container_environs,
                    local_env=Environment(
                        name=f'__rfm_env_{part_name}',
                        modules=site_config.get(f'{partid}/modules'),
                        variables=site_config.get(f'{partid}/variables')
                    ),
                    max_jobs=site_config.get(f'{partid}/max_jobs')
                )
            )

        # Restore configuration, but ignore unresolved sections or
        # configuration parameters at the system level; if we came up to this
        # point, then all is good at the partition level, which is enough.
        site_config.select_subconfig(config_save, ignore_resolve_errors=True)
        return System(
            name=sysname,
            descr=site_config.get('systems/0/descr'),
            hostnames=site_config.get('systems/0/hostnames'),
            modules_system=site_config.get('systems/0/modules_system'),
            preload_env=Environment(
                name=f'__rfm_env_{sysname}',
                modules=site_config.get('systems/0/modules'),
                variables=site_config.get('systems/0/variables')
            ),
            prefix=site_config.get('systems/0/prefix'),
            outputdir=site_config.get('systems/0/outputdir'),
            resourcesdir=site_config.get('systems/0/resourcesdir'),
            stagedir=site_config.get('systems/0/stagedir'),
            partitions=partitions
        )

    @property
    def name(self):
        '''The name of this system.

        :type: :class:`str`
        '''
        return self._name

    @property
    def descr(self):
        '''The description of this system.

        :type: :class:`str`
        '''
        return self._descr

    @property
    def hostnames(self):
        '''The hostname patterns associated with this system.

        :type: :class:`List[str]`
        '''
        return self._hostnames

    @property
    def modules_system(self):
        '''The modules system name associated with this system.

        :type: :class:`reframe.core.modules.ModulesSystem`
        '''
        return self._modules_system

    @property
    def preload_environ(self):
        '''The environment to load whenever ReFrame runs on this system.

        .. versionadded:: 2.19

        :type: :class:`reframe.core.environments.Environment`
        '''
        return self._preload_env

    @property
    def prefix(self):
        '''The ReFrame prefix associated with this system.

        :type: :class:`str`
        '''
        return self._prefix

    @property
    def stagedir(self):
        '''The ReFrame stage directory prefix associated with this system.

        :type: :class:`str`
        '''
        return self._stagedir

    @property
    def outputdir(self):
        '''The ReFrame output directory prefix associated with this system.

        :type: :class:`str`
        '''
        return self._outputdir

    @property
    def resourcesdir(self):
        '''Global resources directory for this system.

        This directory may be used for storing large files related to
        regression tests. The value of this directory is controlled by the
        `resourcesdir <config_reference.html#.systems[].resourcesdir>`__
        configuration parameter.

        :type: :class:`str`

        '''
        return self._resourcesdir

    @property
    def partitions(self):
        '''The system partitions associated with this system.

        :type: :class:`List[SystemPartition]`
        '''
        return utility.SequenceView(self._partitions)

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented

        return (self._name == other._name and
                self._hostnames  == other._hostnames and
                self._partitions == other._partitions)

    def json(self):
        '''Return a JSON object representing this system.'''

        return {
            'name': self._name,
            'descr': self._descr,
            'hostnames': self._hostnames,
            'modules_system': self._modules_system.name,
            'modules': [m for m in self._preload_env.modules],
            'variables': [
                [name, value]
                for name, value in self._preload_env.variables.items()
            ],
            'prefix': self._prefix,
            'outputdir': self._outputdir,
            'stagedir': self._stagedir,
            'resourcesdir': self._resourcesdir,
            'partitions': [p.json() for p in self._partitions]
        }

    def __str__(self):
        return json.dumps(self.json(), indent=2)

    def __repr__(self):
        return (
            f'{type(self).__name__}( '
            f'name={self._name!r}, descr={self._descr!r}, '
            f'hostnames={self._hostnames!r}, '
            f'modules_system={self.modules_system.name!r}, '
            f'preload_env={self._preload_env!r}, prefix={self._prefix!r}, '
            f'outputdir={self._outputdir!r}, '
            f'resourcesdir={self._resourcesdir!r}, '
            f'stagedir={self._stagedir!r}, partitions={self._partitions!r})'
        )
