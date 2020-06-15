# Copyright 2016-2020 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import ipcmagic
import ipyparallel as ipp


get_ipython().run_line_magic('ipcluster', '--version')


get_ipython().run_line_magic('ipcluster', 'start -n 2 --mpi')
# Repeat a few of times in case of `TimeoutError`.
# After the cluser starts, the following calls won't do nothing
# but printing "IPCluster is already running".
# This mimics what the user would do in such case.
get_ipython().run_line_magic('ipcluster', 'start -n 2 --mpi')
get_ipython().run_line_magic('ipcluster', 'start -n 2 --mpi')
get_ipython().run_line_magic('ipcluster', 'start -n 2 --mpi')
get_ipython().run_line_magic('ipcluster', 'start -n 2 --mpi')

c = ipp.Client()

print('cluster ids:', c.ids)

get_ipython().run_cell_magic('px', '', 'import os\nprint(os.popen("ps -u $USER | grep ip").read())')

get_ipython().run_cell_magic('px', '', 'import socket\nsocket.gethostname()')

get_ipython().run_cell_magic('px', '', 'import numpy as np\nimport tensorflow as tf\nimport horovod.tensorflow as hvd')

get_ipython().run_cell_magic('px', '', 'hvd.init()')

get_ipython().run_cell_magic('px', '', '# Note that the generated rando data is different from one node to the other\nnsamples = 1000\nref_slope = 2.0\nref_offset = 0.0\nnoise = np.random.random((nsamples, 1)) - 0.5\nx_train = np.random.random((nsamples, 1)) - 0.5\ny_train = ref_slope * x_train + ref_offset + noise')

get_ipython().run_cell_magic('px', '', '#input pipeline\ndataset = tf.data.Dataset.from_tensor_slices((x_train.astype(np.float32),\n                                              y_train.astype(np.float32)))\ndataset = dataset.shard(hvd.size(), hvd.rank())\ndataset = dataset.batch(500)\ndataset = dataset.repeat(500)\niterator = dataset.make_one_shot_iterator()\nnext_item = iterator.get_next()')

get_ipython().run_cell_magic('px', '', '# Define the model\nslope = tf.Variable(np.random.randn())\noffset = tf.Variable(np.random.randn())\n\nx, y = next_item  # The model is the continuation of the pipeline\n\ny_hat = slope * x + offset\n\nloss = tf.losses.mean_squared_error(y_hat, y)\n\nopt = tf.train.GradientDescentOptimizer(.5)\ntrain = hvd.DistributedOptimizer(opt).minimize(loss)')

get_ipython().run_cell_magic('px', '', 'hooks = [hvd.BroadcastGlobalVariablesHook(0)]')

get_ipython().run_cell_magic('px', '', "history = []\n\nwith tf.train.MonitoredTrainingSession(hooks=hooks) as sess:\n    # Initialization of the variables `slope` and `offset`\n    # is done automatically by tf.train.MonitoredTrainingSession\n    print('rank', hvd.rank(),\n          'inital slope   = %12.6f\\n       initial offset = %12.6f' %\n          sess.run((slope, offset)))\n    while not sess.should_stop():\n        _, loss_val, m, n = sess.run((train, loss, slope, offset))\n        history.append([sess.run(slope), sess.run(offset), loss_val])")

get_ipython().run_cell_magic('px', '', "print('slope=%f   offset=%f   loss=%f' % tuple(history[-1]))")

get_ipython().run_line_magic('ipcluster', 'stop')
