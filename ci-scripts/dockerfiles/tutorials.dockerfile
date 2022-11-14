#
# Execute this from the top-level ReFrame source directory
#


FROM reframehpc/rfm-ci-base:lmod

ENV _SPACK_VER=0.16
ENV _EB_VER=4.4.1

# Required utilities
RUN apt-get -y update && \
    apt-get -y install curl

# ReFrame user
RUN useradd -ms /bin/bash rfmuser

USER rfmuser

# Install Spack
RUN git clone https://github.com/spack/spack ~/spack && \
    cd ~/spack && \
    git checkout releases/v${_SPACK_VER}

RUN pip3 install easybuild==${_EB_VER}

ENV PATH="/home/rfmuser/.local/bin:${PATH}"

# Install ReFrame from the current directory
COPY --chown=rfmuser . /home/rfmuser/reframe/

WORKDIR /home/rfmuser/reframe

RUN ./bootstrap.sh

RUN echo '. /usr/local/lmod/lmod/init/profile && . /home/rfmuser/spack/share/spack/setup-env.sh' > /home/rfmuser/setup.sh

ENV BASH_ENV /home/rfmuser/setup.sh

CMD ["/bin/bash", "-c", "./bin/reframe -r -C tutorials/config/tresa.py -R -c tutorials/build_systems --system tutorials-docker"]
