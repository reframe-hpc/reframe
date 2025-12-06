#
# Execute this from the top-level ReFrame source directory
#


FROM ubuntu:24.04

ENV _SPACK_VER=1.1.0

# Install ReFrame unit test requirements
RUN apt-get -y update && \
    apt-get -y install gcc git make python3 python3-pip

# ReFrame user
RUN useradd -ms /bin/bash rfmuser

USER rfmuser

# Install Spack
RUN git clone --branch v${_SPACK_VER} https://github.com/spack/spack ~/spack

# Install ReFrame from the current directory
COPY --chown=rfmuser . /home/rfmuser/reframe/

WORKDIR /home/rfmuser/reframe

RUN ./bootstrap.sh
RUN pip install --break-system-packages coverage

RUN echo '. /home/rfmuser/spack/share/spack/setup-env.sh' >> /home/rfmuser/.profile
ENV BASH_ENV=/home/rfmuser/.profile

CMD ["/bin/bash", "-c", "coverage run --source=reframe ./test_reframe.py -v --rfm-user-config=ci-scripts/configs/spack.py; coverage xml -o coverage.xml"]
