import reframe as rfm
from hpctestlib.sciapps.gromacs.benchmarks import gromacs_check


@rfm.simple_test
class gromacs_containerized_test(gromacs_check):
    # gromacs_image = variable(str, type(None), value=None)
    gromacs_image = parameter([
        None,
        'nvcr.io/hpc/gromacs:2020',
        'nvcr.io/hpc/gromacs:2020.2',
        'nvcr.io/hpc/gromacs:2021',
        'nvcr.io/hpc/gromacs:2021.3',
        'nvcr.io/hpc/gromacs:2022.1'
    ])
    valid_systems = ['daint:gpu']
    valid_prog_environs = ['gnu']
    use_multithreading = False
    executable = 'gmx mdrun'
    executable_opts += ['-dlb yes', '-ntomp 12', '-npme -1', '-v']
    num_tasks = 1
    num_tasks_per_node = 1
    num_cpus_per_task = 12

    @run_after('init')
    def setup_container_run(self):
        exec_cmd = ' '.join([self.executable, *self.executable_opts])
        self.container_platform.image = self.gromacs_image
        self.container_platform.command = exec_cmd

        if self.gromacs_image is None:
            self.modules = ['GROMACS']
