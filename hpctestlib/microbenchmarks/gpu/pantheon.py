# Copyright 2016-2026 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import glob
import json
import os

import reframe as rfm
import reframe.utility.sanity as sn
import reframe.utility.typecheck as typ


@rfm.simple_test
class pantheon_gpu_check(rfm.RunOnlyRegressionTest):
    '''GPU diagnostics with Pantheon.

    `Pantheon <https://pantheongpu.com>`__ is an open-source GPU diagnostics
    suite for NVIDIA (CUDA) and AMD (ROCm/HIP) cards. Each workload loads one
    part of the card: memory bandwidth, memory cells, memory retention or
    compute. Pantheon verifies the results of every run for silent data
    corruption, and reads the card's error counters before and after it.

    Where :class:`gpu_burn_check` answers whether a card survives a sustained
    load, this test answers whether each part of the card gives correct
    results and how fast, so a degraded card shows up as a failed sanity check
    or as a performance regression against the reference of its node type.

    The test expects the ``pantheon`` executable in the ``PATH``, as installed
    by ``pip install pantheon-gpu``, together with the compiler of the GPU
    toolchain (``nvcc`` or ``hipcc``). Pantheon compiles its workloads on the
    node, for the card it finds there, the first time it runs.

    The sanity check passes if every selected GPU completed the workload,
    Pantheon's own verification found no errors, and the card reported no
    uncorrectable errors during the run.

    .. list-table:: Summary
       :widths: 10 10 20 20 20 20
       :header-rows: 1

       * - Variables
         - Parameters
         - Metrics
         - Fixtures
         - System features
         - Environment features
       * - - :attr:`duration`
           - :attr:`devices`
           - :attr:`mem_percent`
           - :attr:`platform`
         - - :attr:`workload`
         - - :obj:`gpu<N>_score`
           - :obj:`gpu<N>_max_temp`
         - *None*
         - ``+gpu``
         - ``+cuda`` or ``+hip``
    '''

    #: The Pantheon workload to run.
    #:
    #: ``memory_read`` measures memory bandwidth, ``march_test`` tests the
    #: memory cells with a March C- pattern, ``memory_retention`` tests that
    #: they hold their charge, and ``tensor_virus`` loads the FP16 compute
    #: path.
    workload = parameter(['memory_read', 'march_test',
                          'memory_retention', 'tensor_virus'])

    #: Duration of each workload in seconds.
    #:
    #: :type: :class:`int`
    #: :default: ``60``
    duration = variable(int, value=60)

    #: List of device IDs to run the workload on.
    #:
    #: If empty, the workload will run on all the available devices.
    #:
    #: :type: :class:`List[int]`
    #: :default: ``[]``
    devices = variable(typ.List[int], value=[])

    #: Percentage of the free device memory that the workload may use.
    #:
    #: :type: :class:`int`
    #: :default: ``99``
    mem_percent = variable(int, value=99)

    #: The backend to run on: ``'auto'``, ``'cuda'``, ``'hip'`` or ``'mock'``.
    #:
    #: With ``'auto'`` Pantheon detects the backend. ``'mock'`` runs the
    #: workloads on the CPU and tests no hardware; it is there for trying the
    #: test on a system without GPUs.
    #:
    #: :type: :class:`str`
    #: :default: ``'auto'``
    platform = variable(str, value='auto')

    descr = 'GPU diagnostics with Pantheon'
    valid_systems = ['+gpu']
    valid_prog_environs = ['+cuda', '+hip']
    executable = 'pantheon'
    exclusive_access = True
    num_tasks = 1
    num_tasks_per_node = 1
    maintainers = ['saqibkh']
    tags = {'diagnostic', 'gpu'}

    @run_before('run')
    def set_gpus_per_node(self):
        if self.num_gpus_per_node is not None:
            return

        if self.devices:
            self.num_gpus_per_node = len(self.devices)
        else:
            gpus = self.current_partition.select_devices('gpu')
            if gpus:
                self.num_gpus_per_node = gpus[0].num_devices

    @run_before('run')
    def set_exec_opts(self):
        gpus = 'all'
        if self.devices:
            gpus = ','.join(str(d) for d in self.devices)

        self.executable_opts = [
            '--test', self.workload,
            '--duration', str(self.duration),
            '--gpu', gpus,
            '--mem', str(self.mem_percent),
            '--platform', self.platform
        ]

    def report_rows(self):
        '''The result rows of the run, one per GPU.

        Pantheon writes its reports in the ``database`` directory of the
        working directory. A report carries the result row of each GPU and
        the GPUs that were found on the node. A workload that completed is
        reported twice with the same row, so the rows are keyed by GPU.
        '''
        rows = {}
        pattern = os.path.join(self.stagedir, 'database', '*.json')
        for path in sorted(glob.glob(pattern)):
            with open(path) as fp:
                report = json.load(fp)

            kinds = {g.get('id'): g.get('type')
                     for g in report.get('gpu_static_info', [])}
            for row in report.get('test_results', []):
                if row.get('Test Name') == self.workload:
                    rows[row.get('GPU ID')] = dict(
                        row, kind=kinds.get(row.get('GPU ID'))
                    )

        return rows

    @sanity_function
    def validate_run(self):
        rows = self.report_rows()
        on_hardware = [r for r in rows.values() if r['kind'] != 'MOCK']
        checks = [
            sn.assert_found(r'FINAL SUMMARY REPORT', self.stdout),
            sn.assert_found(r'Verification: PASS', self.stdout),
            sn.assert_not_found(r'Verification: FAIL', self.stdout),
            sn.assert_true(rows, msg='Pantheon wrote no report'),
        ]
        if self.devices:
            checks.append(sn.assert_eq(
                sorted(rows), sorted(self.devices),
                msg='not every requested GPU has a result'
            ))

        if self.platform != 'mock':
            checks.append(sn.assert_eq(
                len(on_hardware), len(rows),
                msg=('Pantheon found no GPU with a compiler and fell back '
                     'to its CPU backend: no hardware was tested')
            ))

        for gpu, row in sorted(rows.items()):
            checks += [
                sn.assert_ne(row.get('Unit'), 'ERR',
                             msg=f'GPU {gpu}: the workload failed'),
                sn.assert_gt(row.get('Score', 0), 0,
                             msg=f'GPU {gpu}: no throughput was measured'),
                sn.assert_ne(str(row.get('RAS Status')).upper(), 'ERROR',
                             msg=(f'GPU {gpu}: uncorrectable errors: '
                                  f'{row.get("RAS Error Delta")}'))
            ]

        return sn.all(checks)

    @run_before('performance')
    def set_perf_variables(self):
        '''One throughput and one temperature metric per GPU.

        The unit of the score depends on the workload: GB/s for the memory
        bandwidth, operations per second for the memory tests and TFLOPS for
        the compute workload.
        '''

        self.perf_variables = {}
        for gpu, row in sorted(self.report_rows().items()):
            self.perf_variables[f'gpu{gpu}_score'] = (
                sn.make_performance_function(
                    sn.defer(float(row['Score'])), row.get('Unit') or ''
                )
            )
            temp = row.get('Max Temp (C)')
            if isinstance(temp, (int, float)):
                self.perf_variables[f'gpu{gpu}_max_temp'] = (
                    sn.make_performance_function(sn.defer(float(temp)), 'C')
                )
