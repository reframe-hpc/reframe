#
# Execute this from the top-level ReFrame source directory
#


FROM reframehpc/rfm-ci-base:lmod77

# Install ReFrame from the current directory
COPY --chown=rfmuser . /home/rfmuser/reframe/

WORKDIR /home/rfmuser/reframe

RUN ./bootstrap.sh

CMD ["/bin/bash", "-c", "./test_reframe.py --rfm-user-config=ci-scripts/configs/lmod.py -v"]
