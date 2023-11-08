#
# Execute this from the top-level ReFrame source directory
#

FROM ghcr.io/reframe-hpc/tmod:4.6.0


# ReFrame requirements
RUN \
    apt-get -y update && \
    apt-get -y install gcc make git python3 python3-pip curl

# ReFrame user
RUN useradd -ms /bin/bash rfmuser

USER rfmuser

# Install ReFrame from the current directory
COPY --chown=rfmuser . /home/rfmuser/reframe/

WORKDIR /home/rfmuser/reframe

RUN ./bootstrap.sh

CMD ["/bin/bash", "-c", "./test_reframe.py --rfm-user-config=ci-scripts/configs/tmod4.py -v"]
