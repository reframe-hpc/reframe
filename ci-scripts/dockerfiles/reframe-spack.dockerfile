# Copyright 2016-2026 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

#
# Execute this from the top-level ReFrame source directory
#


FROM ubuntu:24.04
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

ENV _SPACK_VER=1.1.0

# Install ReFrame unit test requirements
RUN apt-get -y update && \
    apt-get -y install gcc git make python3

# ReFrame user
RUN useradd -ms /bin/bash rfmuser

USER rfmuser

# Install Spack
RUN git clone --branch v${_SPACK_VER} https://github.com/spack/spack ~/spack

# Install ReFrame from the current directory
COPY --chown=rfmuser . /home/rfmuser/reframe/

WORKDIR /home/rfmuser/reframe

RUN uv sync --group dev && \
    echo '. /home/rfmuser/spack/share/spack/setup-env.sh' >> /home/rfmuser/.profile
ENV BASH_ENV=/home/rfmuser/.profile

CMD ["/bin/bash", "-c", "uv run coverage run --source=reframe ./test_reframe.py -v --rfm-user-config=ci-scripts/configs/spack.py; uv run coverage xml -o coverage.xml"]
