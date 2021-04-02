#
# Execute this from the top-level ReFrame source directory
#

FROM reframehpc/rfm-ci-base:tmod32

# Install ReFrame from the current directory
COPY --chown=rfmuser . /home/rfmuser/reframe/

WORKDIR /home/rfmuser/reframe

RUN ./bootstrap.sh

CMD ["/bin/bash", "-c", "./test_reframe.py --rfm-user-config=ci-scripts/configs/tmod32.py -v"]
