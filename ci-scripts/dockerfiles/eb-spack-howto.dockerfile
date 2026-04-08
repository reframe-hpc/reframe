# Copyright 2016-2026 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

#
# Execute this from the top-level ReFrame source directory
#


FROM ghcr.io/reframe-hpc/lmod:9.0.4
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

ENV _SPACK_VER=1.1.0
ENV _EB_VER=5.1.2


# Install ReFrame unit test requirements
RUN apt-get -y update && \
    apt-get -y install gcc git make python3 python3-pip curl

# ReFrame user
RUN useradd -ms /bin/bash rfmuser

USER rfmuser

# Install Spack
RUN git clone --branch v${_SPACK_VER} --depth 1 https://github.com/spack/spack ~/spack

ENV PATH="/home/rfmuser/.local/bin:${PATH}"

# Install ReFrame from the current directory
COPY --chown=rfmuser . /home/rfmuser/reframe/

WORKDIR /home/rfmuser/reframe

RUN uv sync --group dev && \
    echo '. /usr/local/lmod/lmod/init/profile && . /home/rfmuser/spack/share/spack/setup-env.sh' >> /home/rfmuser/.profile

# Install EasyBuild
RUN uv pip install easybuild==${_EB_VER}

ENV BASH_ENV=/home/rfmuser/.profile

CMD ["/bin/bash", "-c", "uv run reframe --system=tutorialsys --exec-policy=serial -r -C examples/tutorial/config/baseline_modules.py -R -c examples/tutorial/easybuild/eb_test.py -c examples/tutorial/spack/spack_test.py"]
