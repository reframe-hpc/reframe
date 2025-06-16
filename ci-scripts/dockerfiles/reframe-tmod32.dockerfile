#
# Execute this from the top-level ReFrame source directory
#

FROM ghcr.io/reframe-hpc/tmod:3.2.10

# ReFrame requirements
RUN yum -y install gcc make git python3 python3-pip

# ReFrame user
RUN useradd -ms /bin/bash rfmuser
RUN pip3 install pytest-cov

USER rfmuser

# Install ReFrame from the current directory
COPY --chown=rfmuser . /home/rfmuser/reframe/

WORKDIR /home/rfmuser/reframe

RUN ./bootstrap.sh

CMD ["/bin/bash", "-c", "./test_reframe.py --cov=reframe --cov-report=xml --rfm-user-config=ci-scripts/configs/tmod32.py"]
