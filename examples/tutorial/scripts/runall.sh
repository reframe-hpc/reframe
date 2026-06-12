#!/bin/bash

set -xe

export RFM_ENABLE_RESULTS_STORAGE=1

pushd reframe-examples/tutorial
reframe -c stream/stream_runonly.py -r
reframe -c stream/stream_runonly.py -r
reframe -C config/baseline.py -c stream/stream_runonly.py -r
reframe -C config/baseline_environs.py -c stream/stream_build_run.py --exec-policy=serial -r
reframe -C config/baseline_environs.py -c stream/stream_fixtures.py -l
reframe -C config/baseline_environs.py -c stream/stream_fixtures.py -r
reframe -C config/baseline_environs.py -c stream/stream_variables.py -S num_threads=2 -r
reframe -C config/baseline_environs.py -c stream/stream_variables_fixtures.py --exec-policy=serial -S stream_test.stream_binary.array_size=50000000 -r
reframe -C config/baseline_environs.py -c stream/stream_parameters.py --exec-policy=serial -r
reframe -C config/baseline_environs.py -c stream/stream_variables_fixtures.py -P num_threads=1,2,4,8 --exec-policy=serial -r
reframe -c deps/deps_complex.py -r
reframe --restore-session --failed -r
reframe -c deps/deps_complex.py --keep-stage-files -r
reframe --restore-session --keep-stage-files -n T6 -r
reframe -c deps/deps_complex.py -n T6 -r
popd
