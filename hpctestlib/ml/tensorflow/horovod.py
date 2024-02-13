# Copyright 2016-2024 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe as rfm
import reframe.utility.sanity as sn


@rfm.simple_test
class tensorflow_cnn_check(rfm.RunOnlyRegressionTest, pin_prefix=True):
    '''Run a synthetic CNN benchmark with TensorFlow2 and Horovod.

    TensorFlow is an end-to-end open source platform for machine learning. It
    has a comprehensive, flexible ecosystem of tools, libraries and community
    resources that lets researchers push the state-of-the-art in ML and
    developers easily build and deploy ML powered applications. For more
    information, refer to `<https://www.tensorflow.org/>`__.

    Horovod is a distributed deep learning training framework for TensorFlow,
    Keras, PyTorch, and Apache MXNet. The goal of Horovod is to make
    distributed deep learning fast and easy to use. For more information refer
    to `<https://github.com/horovod/horovod>`__.

    This test runs the Horovod ``tensorflow2_synthentic_benchmark.py``
    example, checks its sanity and extracts the GPU performance.
    '''

    #: The version of Horovod to use.
    #:
    #: :type: :class:`str`
    #: :default: ``'v0.21.0'``
    benchmark_version = variable(str, value='v0.21.0')

    #: The name of the model to use for this benchmark.
    #:
    #: :type: :class:`str`
    #: :default: ``'InceptionV3'``
    model = variable(str, value='InceptionV3')

    #: The size of the batch used during the learning of models.
    #:
    #: :type: :class:`int`
    #: :default: ``32``
    batch_size = variable(int, value=32)

    #: The number of iterations.
    #:
    #: :type: :class:`int`
    #: :default: ``5``
    num_iters = variable(int, value=5)

    #: The number of batches per iteration.
    #:
    #: :type: :class:`int`
    #: :default: ``5``
    num_batches_per_iter = variable(int, value=5)

    #: The number of warmup batches
    #:
    #: :type: :class:`int`
    #: :default: ``5``
    num_warmup_batches = variable(int, value=5)

    executable = 'python tensorflow2_synthetic_benchmark.py'
    tags = {'ml', 'cnn', 'horovod'}

    @run_after('init')
    def prepare_test(self):
        # Get the python script
        script = self.executable.split()[1]

        self.descr = (f'Distributed CNN training with TensorFlow2 and Horovod '
                      f'(model: {self.model})')
        self.prerun_cmds = [
            f'curl -LJO https://raw.githubusercontent.com/horovod/horovod/{self.benchmark_version}/examples/tensorflow2/{script}',  # noqa: E501
            f'sed -i "s/weights=None/weights=None, input_shape=(224, 224, 3)/g" {script}'   # noqa: E501
        ]
        self.executable_opts = [
            f'--model {self.model}',
            f'--batch-size {self.batch_size}',
            f'--num-iters {self.num_iters}',
            f'--num-batches-per-iter {self.num_batches_per_iter}',
            f'--num-warmup-batches {self.num_warmup_batches}'
        ]

    @performance_function('images/s')
    def throughput_iteration(self):
        '''The average GPU throughput per iteration in ``images/s``.'''
        return sn.avg(
            sn.extractall(r'Img/sec per GPU: (\S+) \S+', self.stdout, 1, float)
        )

    @performance_function('images/s')
    def throughput_total(self):
        '''The total GPU throughput of the benchmark in ``images/s``.'''
        return sn.extractsingle(
            rf'Total img/sec on {self.num_tasks} GPU\(s\): (\S+) \S+',
            self.stdout, 1, float
        )

    @sanity_function
    def validate_run(self):
        return sn.all([
            sn.assert_found(rf'Model: {self.model}', self.stdout),
            sn.assert_found(rf'Batch size: {self.batch_size}', self.stdout)
        ])
