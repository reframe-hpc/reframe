#
# Execute this from the top-level ReFrame source directory
#


FROM ghcr.io/reframe-hpc/lmod:8.4.12

ENV _SPACK_VER=0.16
ENV _EB_VER=4.4.1


# Install ReFrame unit test requirements
RUN apt-get -y update && \
    apt-get -y install gcc git make python3 python3-pip

# ReFrame user
RUN useradd -ms /bin/bash rfmuser

USER rfmuser

# Install Spack
RUN git clone --branch releases/v${_SPACK_VER} https://github.com/spack/spack ~/spack && \
    cd ~/spack

RUN pip3 install easybuild==${_EB_VER}

ENV PATH="/home/rfmuser/.local/bin:${PATH}"

# Install ReFrame from the current directory
COPY --chown=rfmuser . /home/rfmuser/reframe/

WORKDIR /home/rfmuser/reframe

RUN ./bootstrap.sh

RUN echo '. /usr/local/lmod/lmod/init/profile && . /home/rfmuser/spack/share/spack/setup-env.sh' > /home/rfmuser/setup.sh

ENV BASH_ENV /home/rfmuser/setup.sh

CMD ["/bin/bash", "-c", "./bin/reframe --system=tutorialsys -r -C examples/tutorial/config/baseline_modules.py -R -c examples/tutorial/easybuild/eb_test.py -c examples/tutorial/spack/spack_test.py"]
