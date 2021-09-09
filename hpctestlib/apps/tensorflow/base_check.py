# Copyright 2016-2021 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe as rfm
import reframe.utility.sanity as sn


class TensorFlow2Horovod_BaseTest(rfm.RunOnlyRegressionTest):
    '''Base class for the TensorFlow2 Horovod Test.

    TensorFlow is an end-to-end open source platform for machine
    learning. It has a comprehensive, flexible ecosystem of tools,
    libraries and community resources that lets researchers push the
    state-of-the-art in ML and developers easily build and deploy ML
    powered applications. (see tensorflow.org).


    Horovod is a distributed deep learning training
    framework for TensorFlow, Keras, PyTorch, and Apache
    MXNet. The goal of Horovod is to make distributed deep
    learning fast and easy to use (see github.com/horovod/horovod).

    This test tests the performance of TensorFlow2 and Horovod using
    classic deep learning model Inception v3. It checks whether learning is
    performed to the end. The default assumption
    is that TensorFlow2 and Horovod is already installed on the device
    under test.
    '''

    descr = 'Distributed training with TensorFlow2 and Horovod'

    #: Name of the model, that used for the testing
    #:
    #: :type: str
    model = 'InceptionV3'

    #: Size of the batch, that used during the learning of models
    #:
    #: :type: int
    #: :default: 32
    batch_size = variable(int, value=32)

    #: :default: :class:`required`
    executable = required

    executable = 'python'

    #: Executable script for the tests
    #:
    #: :type: str
    script = 'tensorflow2_synthetic_benchmark.py'

    @run_after('init')
    def set_prerun_cmds(self):
        self.prerun_cmds = ['wget https://raw.githubusercontent.com/horovod/'
                            'horovod/842d1075e8440f15e84364f494645c28bf20c3ae/'
                            'examples/tensorflow2_synthetic_benchmark.py',
                            'sed -i "s/weights=None/weights=None, '
                            f'input_shape=(224, 224, 3)/g" {self.script}']

    @performance_function('images/s', perf_key='throughput_per_gpu')
    def set_perf_per_gpu(self):
        return sn.extractsingle(
            r'Img/sec per GPU: (?P<throughput_per_gpu>\S+) \S+',
            self.stdout, 'throughput_per_gpu', float)

    @performance_function('images/s', perf_key='throughput')
    def set_perf(self):
        return sn.extractsingle(
            rf'Total img/sec on {self.num_tasks} GPU\(s\): '
            rf'(?P<throughput>\S+) \S+',
            self.stdout, 'throughput', float)

    @sanity_function
    def assert_energy_readout(self):
        return sn.all([
            sn.assert_found(rf'Model: {self.model}', self.stdout),
            sn.assert_found(rf'Batch size: {self.batch_size}', self.stdout)
        ])
