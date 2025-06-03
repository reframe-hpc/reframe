# Copyright 2016-2025 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

FROM ghcr.io/reframe-hpc/reframe-pseudo-cluster-base:22.04

COPY .github/pseudo-cluster/reframe/docker-entrypoint.sh /usr/bin/

RUN chmod +x /usr/bin/docker-entrypoint.sh

WORKDIR /home/admin

COPY . /usr/local/share/reframe

ENV USER admin
ENV SHELL bash

ENTRYPOINT ["/usr/bin/docker-entrypoint.sh"]
