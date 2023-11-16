#
# Execute this from the top-level ReFrame source directory
#


FROM ghcr.io/reframe-hpc/lmod:7.7

# Install ReFrame unit test requirements
RUN apt-get -y update && \
    apt-get -y install gcc make python3 python3-pip curl

# ReFrame user
RUN useradd -ms /bin/bash rfmuser

USER rfmuser

# Install ReFrame from the current directory
COPY --chown=rfmuser . /home/rfmuser/reframe/

WORKDIR /home/rfmuser/reframe

RUN ./bootstrap.sh

CMD ["/bin/bash", "-c", "./test_reframe.py --rfm-user-config=ci-scripts/configs/lmod.py -v"]
