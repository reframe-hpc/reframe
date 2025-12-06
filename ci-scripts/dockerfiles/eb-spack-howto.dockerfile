#
# Execute this from the top-level ReFrame source directory
#


FROM ghcr.io/reframe-hpc/lmod:9.0.4

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
RUN pip3 install --break-system-packages easybuild==${_EB_VER}

ENV PATH="/home/rfmuser/.local/bin:${PATH}"

# Install ReFrame from the current directory
COPY --chown=rfmuser . /home/rfmuser/reframe/

WORKDIR /home/rfmuser/reframe

RUN ./bootstrap.sh

RUN echo '. /usr/local/lmod/lmod/init/profile && . /home/rfmuser/spack/share/spack/setup-env.sh' > /home/rfmuser/setup.sh

ENV BASH_ENV=/home/rfmuser/setup.sh

CMD ["/bin/bash", "-c", "./bin/reframe --system=tutorialsys --exec-policy=serial -r -C examples/tutorial/config/baseline_modules.py -R -c examples/tutorial/easybuild/eb_test.py -c examples/tutorial/spack/spack_test.py"]
