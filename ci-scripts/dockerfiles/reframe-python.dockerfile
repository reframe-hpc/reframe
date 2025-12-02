# Copyright 2016-2025 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

ARG PYTHON_VERSION=3.9

FROM docker.io/python:${PYTHON_VERSION}

# ReFrame user
RUN useradd -ms /bin/bash rfmuser

USER rfmuser

# Install ReFrame from the current directory
COPY --chown=rfmuser . /home/rfmuser/reframe/

WORKDIR /home/rfmuser/reframe

RUN ./bootstrap.sh +docs

CMD ["/bin/bash", "-c", "./test_reframe.py --cov=reframe --cov-report=xml"]
