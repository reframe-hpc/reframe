# Copyright 2016-2020 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import ipcmagic
import ipyparallel as ipp


get_ipython().run_line_magic('ipcluster', '--version')

get_ipython().run_line_magic('ipcluster', 'start -n 2 --mpi')
# Repeat a few times in case of `TimeoutError`.
# After the cluser starts, the following calls won't do anything
# but printing "IPCluster is already running".
# This mimics what the user would do in such case.
get_ipython().run_line_magic('ipcluster', 'start -n 2 --mpi')
get_ipython().run_line_magic('ipcluster', 'start -n 2 --mpi')
get_ipython().run_line_magic('ipcluster', 'start -n 2 --mpi')
get_ipython().run_line_magic('ipcluster', 'start -n 2 --mpi')


c = ipp.Client()

print('cluster ids:', c.ids)

get_ipython().run_cell_magic('px', '',
"""
import os
import socket


print(os.popen("ps -u $USER | grep ip").read())
socket.gethostname()
""")

get_ipython().run_cell_magic('px', '',
"""
import numpy as np
import tensorflow as tf
import horovod.tensorflow.keras as hvd
""")

get_ipython().run_cell_magic('px', '', 'hvd.init()')

get_ipython().run_cell_magic('px', '',
"""
# Create a linear function with noise as our data
nsamples = 1000
ref_slope = 2.0
ref_offset = 0.0
noise = np.random.random((nsamples, 1)) - 0.5    # -0.5 to center the noise
x_train = np.random.random((nsamples, 1)) - 0.5  # -0.5 to center x around 0
y_train = ref_slope * x_train + ref_offset + noise
""")

get_ipython().run_cell_magic('px', '',
"""
dataset = tf.data.Dataset.from_tensor_slices((x_train.astype(np.float32),
                                              y_train.astype(np.float32)))
dataset = dataset.shuffle(1000)
dataset = dataset.batch(100)
dataset = dataset.repeat(500)
""")

get_ipython().run_cell_magic('px', '',
"""
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(1, input_shape=(1,),
    activation='linear')])

opt = tf.keras.optimizers.SGD(lr=0.5)
opt = hvd.DistributedOptimizer(opt)

model.compile(optimizer=opt, loss='mse')
""")

get_ipython().run_cell_magic('px', '',
"""
class TrainHistory(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.vars = []
        self.loss = []

    def on_batch_end(self, batch, logs={}):
        self.vars.append([v.numpy() for v in self.model.variables])
        self.loss.append(logs.get('loss'))

history = TrainHistory()
""")

get_ipython().run_cell_magic('px', '',
'initial_sync = hvd.callbacks.BroadcastGlobalVariablesCallback(0)')

get_ipython().run_cell_magic('px', '',
'fit = model.fit(dataset, callbacks=[initial_sync, history])')

get_ipython().run_cell_magic('px', '',
"""print(f'slope={history.vars[-1][0][0][0]}   '
         f'offset={history.vars[-1][1][0]}  '
         f' loss={history.loss[-1]}')
""")

get_ipython().run_line_magic('ipcluster', 'stop')
