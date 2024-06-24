#
# Execute this from the top-level ReFrame source directory
#


FROM ghcr.io/reframe-hpc/lmod:8.4.12

# Install ReFrame unit test requirements
RUN apt-get -y update && \
    apt-get -y install gcc git make python3 python3-pip

# ReFrame user
RUN useradd -ms /bin/bash rfmuser

USER rfmuser

# Install ReFrame from the current directory
COPY --chown=rfmuser . /home/rfmuser/reframe/

WORKDIR /home/rfmuser/reframe

RUN ./bootstrap.sh
RUN pip install pytest-cov

CMD ["/bin/bash", "-c", "./test_reframe.py --cov=reframe --cov-report=xml --rfm-user-config=ci-scripts/configs/lmod.py"]
