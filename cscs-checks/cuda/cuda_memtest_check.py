import reframe as rfm
import reframe.utility.sanity as sn


@rfm.simple_test
class CudaMemtest(rfm.RegressionTest):
    def __init__(self):
        super().__init__()
        self.valid_systems = ['daint:gpu', 'dom:gpu', 'tiger:gpu']
        self.valid_prog_environs = ['PrgEnv-cray']
        self.descr = 'Flexible CUDA Memtest'
        self.maintainers = ['TM', 'SK']
        self.num_tasks_per_node = 1
        self.num_tasks = 0
        self.num_gpus_per_node = 1
        self.modules = ['cudatoolkit']
        src_url = ('https://downloads.sourceforge.net/project/cudagpumemtest/'
                   'cuda_memtest-1.2.3.tar.gz')
        self.prebuild_cmd = [
            'wget %s' % src_url,
            'tar -xzf cuda_memtest-1.2.3.tar.gz',
            'cd cuda_memtest-1.2.3',
            'patch -p1 < ../cuda_memtest-1.2.3.patch'
        ]
        self.build_system = 'Make'
        self.executable = './cuda_memtest-1.2.3/cuda_memtest'
        self.executable_opts = ['--disable_test', '6', '--num_passes', '1']

        valid_test_ids = {i for i in range(11) if i not in {6, 9}}
        assert_finished_tests = [
            sn.assert_eq(
                sn.count(sn.findall('Test%s finished' % test_id, self.stdout)),
                self.num_tasks_assigned)
            for test_id in valid_test_ids
        ]
        self.sanity_patterns = sn.all([
            *assert_finished_tests,
            sn.assert_not_found('(?i)ERROR', self.stdout),
            sn.assert_not_found('(?i)ERROR', self.stderr)])
        self.tags = {'diagnostic', 'ops', 'craype'}

    @property
    @sn.sanity_function
    def num_tasks_assigned(self):
        return self.job.num_tasks
