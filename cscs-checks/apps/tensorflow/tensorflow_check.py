import reframe as rfm
import reframe.utility.sanity as sn


class TensorFlowBaseTest(rfm.RunOnlyRegressionTest):
    def __init__(self, model_name):
        self.name = 'tensorflow_%s_check' % model_name
        self.descr = 'Tensorflow official %s test' % model_name
        self.valid_systems = ['daint:gpu', 'dom:gpu']
        self.valid_prog_environs = ['PrgEnv-gnu']
        self.sourcesdir = 'https://github.com/tensorflow/models.git'
        self.maintainers = ['TM']
        self.tags = {'production'}
        self.num_tasks = 1
        self.num_gpus_per_node = 1
        tf_version = '1.14.0'
        cuda_version = '10.1.168'
        self.modules = ['TensorFlow/%s-CrayGNU-19.06-cuda-%s-python3' %
                        (tf_version, cuda_version)]

        # Checkout to the branch corresponding to the module version of
        # TensorFlow
        self.pre_run = ['git checkout r%s' % tf_version]
        self.variables = {'PYTHONPATH': '$PYTHONPATH:.'}


@rfm.simple_test
class TensorFlowMnistTest(TensorFlowBaseTest):
    def __init__(self):
        super().__init__('mnist')

        train_epochs = 10
        self.executable = 'python3 ./official/mnist/mnist.py'
        self.executable_opts = ['--model_dir', '.', '--export_dir', '.',
                                ' --data_dir', '.', '--train_epochs',
                                str(train_epochs)]

        self.sanity_patterns = sn.all([
            sn.assert_found(r'Finished evaluation at',
                            self.stderr),
            sn.assert_gt(sn.extractsingle(
                r"Evaluation results:\s+\{.*'accuracy':\s+(?P<accuracy>\S+)"
                r"(?:,|\})", self.stdout, 'accuracy', float, -1), 0.99)
        ])


@rfm.simple_test
class TensorFlowWidedeepTest(TensorFlowBaseTest):
    def __init__(self):
        super().__init__('wide_deep')

        train_epochs = 10
        self.executable = 'python3 ./official/wide_deep/census_main.py'
        self.executable_opts = [
            '--data_dir', './official/wide_deep/',
            '--model_dir', './official/wide_deep/model_dir',
            '--train_epochs', str(train_epochs)]

        self.sanity_patterns = sn.all([
            sn.assert_found(r'Finished evaluation at',
                            self.stderr),
            sn.assert_reference(sn.extractsingle(
                r"Results at epoch %s[\s\S]+accuracy:\s+(?P<accuracy>\S+)" %
                train_epochs, self.stderr, 'accuracy', float, -1),
                0.85, -0.05, None)
        ])

        self.pre_run += ['mkdir ./official/wide_deep/model_dir',
                         'python3 ./official/wide_deep/census_dataset.py '
                         '--data_dir ./official/wide_deep/']
