# Copyright 2016-2026 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

#
# Execute this from the top-level ReFrame source directory
#

FROM ghcr.io/reframe-hpc/envmodules:5.6.1
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

# ReFrame requirements
RUN \
    apt-get -y update && \
    apt-get -y install gcc make git python3

# ReFrame user
RUN useradd -ms /bin/bash rfmuser

USER rfmuser

# Install ReFrame from the current directory
COPY --chown=rfmuser . /home/rfmuser/reframe/

WORKDIR /home/rfmuser/reframe

RUN uv sync --group dev && \
    echo ". $BASH_ENV" >> /home/rfmuser/.profile
ENV BASH_ENV=/home/rfmuser/.profile

CMD ["/bin/bash", "-c", "uv run coverage run --source=reframe ./test_reframe.py --rfm-user-config=ci-scripts/configs/envmod.py; uv run coverage xml -o coverage.xml"]
