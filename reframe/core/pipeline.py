#
# Basic functionality for regression tests
#

import copy
import glob
import os
import logging.config
import shutil

import reframe
import reframe.settings as settings
import reframe.utility.os as os_ext

from reframe.core.environments import Environment
from reframe.core.exceptions import ReframeError
from reframe.core.fields import *
from reframe.core.launchers  import *
from reframe.core.schedulers import *
from reframe.core.shell import BashScriptBuilder
from reframe.core.systems import System, SystemPartition
from reframe.frontend.resources import ResourcesManager


class RegressionTest:
    """Base class for regression checks providing the implementation of the
       different phases the regression goes through"""

    name                = AlphanumericField('name')
    valid_prog_environs = TypedListField('valid_prog_environs', str)
    valid_systems       = TypedListField('valid_systems', str)
    resources           = TypedField('resouces', ResourcesManager)
    descr               = StringField('descr')
    sourcepath          = StringField('sourcepath')
    prebuild_cmd        = TypedListField('prebuild_cmd', str)
    postbuild_cmd       = TypedListField('postbuild_cmd', str)
    executable          = StringField('executable')
    executable_opts     = TypedListField('executable_opts', str)
    current_system      = TypedField('current_system', System)
    current_partition   = TypedField('current_partition', SystemPartition,
                                     allow_none=True)
    current_environ     = TypedField('current_environ', Environment,
                                     allow_none=True)
    keep_files          = TypedListField('keep_files', str)
    readonly_files      = TypedListField('readonly_files', str)
    tags                = TypedSetField('tags', str)
    maintainers         = TypedListField('maintainers', str)
    strict_check        = BooleanField('strict_check')
    num_tasks           = IntegerField('num_tasks')
    num_tasks_per_node  = IntegerField('num_tasks_per_node', allow_none=True)
    num_gpus_per_node   = IntegerField('num_gpus_per_node')
    num_cpus_per_task   = IntegerField('num_cpus_per_task', allow_none=True)
    num_tasks_per_core  = IntegerField('num_tasks_per_core', allow_none=True)
    num_tasks_per_socket = IntegerField('num_tasks_per_socket', allow_none=True)
    use_multithreading  = BooleanField('use_multithreading')
    local               = BooleanField('local')
    prefix              = StringField('prefix')
    sourcesdir          = StringField('sourcesdir')
    stagedir            = StringField('stagedir', allow_none=True)
    stdout              = StringField('stdout', allow_none=True)
    stderr              = StringField('stderr', allow_none=True)
    _logfile            = StringField('_logfile', allow_none=True)
    reference           = ScopedDictField('reference', object)
    sanity_patterns     = SanityPatternField('sanity_patterns', allow_none=True)
    perf_patterns       = SanityPatternField('perf_patterns', allow_none=True)
    modules             = TypedListField('modules', str)
    variables           = TypedDictField('variables', str, str)
    time_limit          = TimerField('time_limit')
    job                 = TypedField('job', Job, allow_none=True)
    job_resources       = TypedDictField('job_resources', str, str)

    def __init__(self, name, prefix, system, resources):
        self.name                = name
        self.valid_prog_environs = []
        self.valid_systems       = []
        self.descr               = name

        self.sourcepath        = ''
        self.prebuild_cmd      = []
        self.postbuild_cmd     = []
        self.executable        = os.path.join('.', self.name)
        self.executable_opts   = []
        self.current_system    = system
        self.current_partition = None
        self.current_environ   = None
        self.job               = None
        self.job_resources     = {}
        self.keep_files        = []
        self.readonly_files    = []
        self.tags              = set()
        self.maintainers       = []

        # Strict performance check, if applicable
        self.strict_check = True

        # Default is a single node check
        self.num_tasks = 1
        self.num_tasks_per_node = None
        self.num_gpus_per_node = 0
        self.num_cpus_per_task = None
        self.num_tasks_per_core = None
        self.num_tasks_per_socket = None
        self.use_multithreading = False

        # True only if check is to be run locally
        self.local = False

        # Static directories of the regression check
        self.prefix        = os.path.abspath(prefix)
        self.sourcesdir    = os.path.join(self.prefix, 'src')

        # Dynamic paths of the regression check; will be set in setup()
        self.stagedir = None
        self.stdout   = None
        self.stderr   = None
        self._logfile = None

        # Output patterns
        self.sanity_patterns = None

        # Performance patterns: None -> no performance checking
        self.perf_patterns = None
        self.reference = {}

        # Environment setup
        self.modules = []
        self.variables = {}

        # Time limit for the check
        self.time_limit = (0, 10, 0)

        # Private fields
        self._resources = resources

        # Compilation task output; not meant to be touched by users
        self._compile_task = None

        # Check-specific logging
        self._logger = None

        # Type of launcher to use for launching jobs
        self._launcher_type = None


    def supports_system(self, partition_name):
        if '*' in self.valid_systems:
            return True

        if self.current_system.name in self.valid_systems:
            return True

        # Check if this is a relative name
        if partition_name.find(':') == -1:
            partition_name = '%s:%s' % (self.current_system.name,
                                        partition_name)

        return partition_name in self.valid_systems


    def supports_progenv(self, env_name):
        if '*' in self.valid_prog_environs:
            return True

        return env_name in self.valid_prog_environs


    def is_local(self):
        return self.local or self.current_partition.scheduler == 'local'


    def _setup_environ(self, environ):
        """Setup the current environment and load it."""

        self.current_environ = environ

        # Add user modules and variables to the environment
        for m in self.modules:
            self.current_environ.add_module(m)

        for k, v in self.variables.items():
            self.current_environ.set_variable(k, v)

        # First load the local environment of the partition
        self.current_partition.local_env.load()
        self.current_environ.load()


    def _setup_paths(self):
        """Setup the check's dynamic paths."""
        self.stagedir  = self._resources.stagedir(
            self.current_partition.name, self.name, self.current_environ.name)
        self.outputdir = self._resources.outputdir(
            self.current_partition.name, self.name, self.current_environ.name)
        self.stdout = os.path.join(self.stagedir, '%s.out' % self.name)
        self.stderr = os.path.join(self.stagedir, '%s.err' % self.name)


    def _setup_job(self, **job_opts):
        """Setup the job related to this check."""

        # num_gpus_per_node is a managed resource
        if self.num_gpus_per_node > 0:
            self.job_resources.setdefault('num_gpus_per_node',
                                          str(self.num_gpus_per_node))

        # If check is local, use the LocalLauncher, otherwise try to infer the
        # launcher from the system info
        if self.is_local():
            self._launcher_type = LocalLauncher
        elif self.current_partition.scheduler == 'nativeslurm':
            self._launcher_type = NativeSlurmLauncher
        elif self.current_partition.scheduler == 'slurm+alps':
            self._launcher_type = AlpsLauncher
        else:
            # Oops
            raise RegressionFatalError('Oops: unsupported launcher: %s' %
                                       self.current_partition.scheduler)

        job_name = '%s_%s_%s_%s' % (self.name,
                                    self.current_system.name,
                                    self.current_partition.name,
                                    self.current_environ.name)
        if self.is_local():
            self.job = LocalJob(
                job_name=job_name,
                job_environ_list=[
                    self.current_partition.local_env,
                    self.current_environ
                ],
                job_script_builder=BashScriptBuilder(),
                stdout=self.stdout,
                stderr=self.stderr,
                time_limit=self.time_limit,
                **job_opts)
        else:
            # We need to deep copy job_opts since we may be called repeatedly
            # from the front-end
            job_opts = copy.deepcopy(job_opts)

            self.job = SlurmJob(
                job_name=job_name,
                job_environ_list=[
                    self.current_partition.local_env,
                    self.current_environ
                ],
                job_script_builder=BashScriptBuilder(login=True),
                num_tasks=self.num_tasks,
                num_tasks_per_node=self.num_tasks_per_node,
                num_cpus_per_task=self.num_cpus_per_task,
                num_tasks_per_core=self.num_tasks_per_core,
                num_tasks_per_socket=self.num_tasks_per_socket,
                use_smt=self.use_multithreading,
                launcher=self._launcher_type,
                stdout=self.stdout,
                stderr=self.stderr,
                time_limit=self.time_limit,
                **job_opts)

            # Get job options from managed resources and prepend them to
            # job_opts. We want any user supplied options to be able to override
            # those set by the framework.
            resources_opts = []
            for r, v in self.job_resources.items():
                resources_opts.extend(self.current_partition.get_resource(r, v))

            self.job.options = self.current_partition.access + \
                               resources_opts + self.job.options

        # Prepend job path to script name
        self.job.script_filename = os.path.join(self.stagedir,
                                                self.job.script_filename)


    # FIXME: This is a temporary solution to address issue #157
    def _setup_logging(self):
        self._logfile = os.path.join(
            self._resources.logdir(self.current_partition.name),
            self.name + '.log'
        )

        self._logger = logging.getLogger('reframe.checks.%s' % self.name)
        formatter = logging.Formatter(
            fmt='[%(asctime)s] %(name)s: %(levelname)s: %(message)s',
            datefmt='%FT%T'
        )

        handler = logging.handlers.RotatingFileHandler(
            filename=self._logfile,
            maxBytes=10*1024*1024,
        )
        handler.setLevel(logging.INFO)
        handler.setFormatter(formatter)

        self._logger.addHandler(handler)
        self._logger.setLevel(logging.INFO)


    def setup(self, system, environ, **job_opts):
        self.current_partition = system
        self._setup_environ(environ)
        self._setup_paths()
        self._setup_job(**job_opts)
        if self.perf_patterns != None:
            self._setup_logging()


    def _copy_to_stagedir(self, path):
        os_ext.copytree_virtual(path, self.stagedir, self.readonly_files)


    def prebuild(self):
        for cmd in self.prebuild_cmd:
            os_ext.run_command(cmd, check=True)


    def postbuild(self):
        for cmd in self.postbuild_cmd:
            os_ext.run_command(cmd, check=True)


    def compile(self, **compile_opts):
        if not self.current_environ:
            raise ReframeError('No programming environment set')

        # if self.sourcepath refers to a directory, stage it first
        target_sourcepath = os.path.join(self.sourcesdir, self.sourcepath)
        if os.path.isdir(target_sourcepath):
            self._copy_to_stagedir(target_sourcepath)
            self.current_environ.include_search_path.append(self.stagedir)
            target_sourcepath = self.stagedir
            includedir = os.path.abspath(self.stagedir)
        else:
            includedir = os.path.abspath(self.sourcesdir)

        # Add the the correct source directory to the include path
        self.current_environ.include_search_path.append(includedir)

        # Remove source and executable from compile_opts
        compile_opts.pop('source', None)
        compile_opts.pop('executable', None)

        # Change working dir to stagedir although absolute paths are used
        # everywhere in the compilation process. This is done to ensure that any
        # other files (besides the executable) generated during the the
        # compilation will remain in the stage directory
        wd_save = os.getcwd()
        os.chdir(self.stagedir)
        try:
            self.prebuild()
            self._compile_task = self.current_environ.compile(
                sourcepath=target_sourcepath,
                executable=os.path.join(self.stagedir, self.executable),
                **compile_opts)
            self.postbuild()
        finally:
            # Always restore working directory
            os.chdir(wd_save)


    def run(self):
        if not self.current_system or not self.current_partition:
            raise ReframeError('No system or system partition is set')

        self.job.submit(cmd='%s %s' %
                        (self.executable, ' '.join(self.executable_opts)),
                        workdir=self.stagedir)
        if self._logger:
            msg = 'submitted job' if not self.is_local() else 'launched process'
            self._logger.info('%s (id=%s)' % (msg, self.job.jobid))


    def wait(self):
        self.job.wait()


    def check_sanity(self):
        return self._match_patterns(self.sanity_patterns, None)


    def check_performance_relaxed(self):
        """Implements the relaxed performance check logic."""
        ret = self.check_performance()
        return ret if self.strict_check else True


    def check_performance(self):
        return self._match_patterns(self.perf_patterns, self.reference)


    def cleanup(self, remove_files=False, unload_env=True):
        # Copy stdout/stderr and job script
        shutil.copy(self.stdout, self.outputdir)
        shutil.copy(self.stderr, self.outputdir)
        if self.job:
            shutil.copy(self.job.script_filename, self.outputdir)

        # Copy files specified by the user
        for f in self.keep_files:
            if not os.path.isabs(f):
                f = os.path.join(self.stagedir, f)
            shutil.copy(f, self.outputdir)

        if remove_files:
            shutil.rmtree(self.stagedir)

        if unload_env:
            self.current_environ.unload()
            self.current_partition.local_env.unload()


    def _match_patterns_file(self, path, patterns, reference):
        def _resolve_tag(tag):
            try:
                return reference['%s:%s:%s' % \
                                 (self.current_system.name,
                                  self.current_partition.name, tag)]
            except KeyError:
                raise ReframeError(
                    "tag `%s' could not be resolved "
                    "in perf. references for `%s'" %
                    (tag, self.current_system.name)
                )

        matched_patt = set()
        found_tags   = set()
        file = None
        try:
            file = open(path, 'rt', encoding='utf-8')
            for line in file:
                for patt, taglist in patterns.items():
                    match = re.search(patt, line)
                    if not match:
                        continue

                    matched_patt.add(patt)
                    for td in taglist:
                        tag, conv, thres = td
                        ref = _resolve_tag(tag) \
                              if reference != None else None
                        if thres(value=conv(match.group(tag)),
                                 reference=ref,
                                 logger=self._logger):
                            found_tags.add(tag)
        except (OSError, ValueError) as e:
            raise ReframeError('Caught %s: %s' % (type(e).__name__, str(e)))
        finally:
            if file:
                file.close()

        return (matched_patt, found_tags)


    def _match_patterns(self, multi_patterns, reference):
        if not multi_patterns:
            return True

        for file_patt, patterns in multi_patterns.items():
            if file_patt == '-' or file_patt == '&1':
                files = [ self.stdout ]
            elif file_patt == '&2':
                files = [ self.stderr ]
            else:
                files = glob.glob(os.path.join(self.stagedir, file_patt))

            if not files:
                # No output files found
                return False

            # Check if an eof handler is present and temporarily remove it from
            # patterns
            if '\e' in patterns.keys():
                eof_handler = patterns['\e']
                del patterns['\e']
            else:
                eof_handler = None

            required_tags = frozenset(
                [ td[0] for taglist in patterns.values() for td in taglist ]
            )

            ret = True
            for filename in files:
                matched_patt, found_tags = self._match_patterns_file(
                    filename, patterns, reference
                )
                if matched_patt != patterns.keys() or \
                   found_tags != required_tags:
                    ret = False

                # We need eof_handler to be called anyway that's why we do not
                # combine this check with the above and we delay the breaking
                # out of the loop here
                if eof_handler and not eof_handler(logger=self._logger):
                    ret = False
                    break

            if eof_handler:
                # Restore the handler
                patterns['\e'] = eof_handler

        return ret


    def __str__(self):
        return '%s (%s)\n' \
                '        tags: [ %s ], maintainers: [ %s ]' % \
                (self.name, self.descr,
                ', '.join(self.tags), ', '.join(self.maintainers))


class RunOnlyRegressionTest(RegressionTest):
    def compile(self, **compile_opts):
        pass


    def run(self):
        self._copy_to_stagedir(os.path.join(self.sourcesdir, self.sourcepath))
        super().run()


class CompileOnlyRegressionTest(RegressionTest):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.local = True


    # No need to setup the job for compile-only checks
    def setup(self, system, environ, **job_opts):
        self.current_partition = system
        self._setup_environ(environ)
        self._setup_paths()


    def compile(self, **compile_opts):
        super().compile(**compile_opts)

        with open(self.stdout, 'w') as f:
            f.write(self._compile_task.stdout)

        with open(self.stderr, 'w') as f:
            f.write(self._compile_task.stderr)


    def run(self):
        pass


    def wait(self):
        pass
