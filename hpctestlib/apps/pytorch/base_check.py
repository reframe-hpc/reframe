# Copyright 2016-2021 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe as rfm
import reframe.utility.sanity as sn


class PytorchHorovod_BaseTest(rfm.RunOnlyRegressionTest):
    '''Base class for the Pytorch Horovod Test.

    PyTorch is an optimized tensor library for deep
    learning using GPUs and CPUs (see pytorch.org).


    Horovod is a distributed deep learning training
    framework for TensorFlow, Keras, PyTorch, and Apache
    MXNet. The goal of Horovod is to make distributed deep
    learning fast and easy to use (see github.com/horovod/horovod).

    This test tests the performance of Pytorch and Horovod using
    two classic deep learning models. It checks whether learning is
    performed to the end. The default assumption
    is that Pytorch and Horovod is already installed on the device
    under test.
    '''

    #: Model of pytorch, that used for the tests
    model = parameter(['inception_v3', 'resnet50'])

    #: :default: :class:`required`
    executable = required

    executable = 'python'

    #: This variables define the way to the source file
    hash = 'master'
    git_url = f'https://raw.githubusercontent.com/horovod/horovod/{hash}/examples/pytorch'  # noqa: E501
    git_src = 'pytorch_synthetic_benchmark.py'

    #: Size of the batch, that used during the learning of models
    #:
    #: :type: int
    #: :default: 32
    batch_size = variable(int, value=32)

    descr = 'Distributed training with Pytorch and Horovod'

    @run_after('init')
    def set_prerun_cmds(self):
        self.prerun_cmds = [f'wget {self.git_url}/{self.git_src}']
        if self.model == 'inception_v3':
            self.prerun_cmds += [
                'python3 -m venv --system-site-packages myvenv',
                'source myvenv/bin/activate',
                'pip install scipy',
                'sed -i "s-output = model(data)-output, aux = model(data)-"'
                f' {self.git_src}',
                'sed -i "s-data = torch.randn(args.batch_size, 3, 224, 224)-'
                f'data = torch.randn(args.batch_size, 3, 299, 299)-"'
                f' {self.git_src}'
            ]

    @run_after('init')
    def set_executable_opts(self):
        self.executable_opts = [
            self.git_src,
            f'--model {self.model}',
            f'--batch-size {self.batch_size}',
            '--num-iters 5',
            '--num-batches-per-iter 5'
        ]

    @performance_function('images/s', perf_key='throughput_per_gpu')
    def set_perf_per_gpu(self):
        return sn.extractsingle(
            r'Img/sec per GPU: (?P<throughput_per_gpu>\S+) \S+',
            self.stdout, 'throughput_per_gpu', float)

    @performance_function('images/s', perf_key='throughput_per_job')
    def set_perf_per_job(self):
        return sn.extractsingle(
            r'Total img/sec on \d+ GPU\(s\): (?P<throughput>\S+) \S+',
            self.stdout, 'throughput', float)

    @sanity_function
    def assert_energy_readout(self):
        return  sn.all([
            sn.assert_found(rf'Model: {self.model}', self.stdout),
            sn.assert_found(rf'Batch size: {self.batch_size}', self.stdout)
        ])
