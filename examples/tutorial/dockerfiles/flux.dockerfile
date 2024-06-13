FROM fluxrm/flux-sched:focal
# docker build -f tutorials/flux/Dockerfile -t flux-reframe .
# docker run -it -v $PWD:/code flux-reframe
# docker run -it flux-reframe
USER root
ENV PATH=/opt/conda/bin:$PATH
WORKDIR /code
COPY . /code
RUN /bin/bash /code/bootstrap.sh && \
    python3 -m pip install pytest
ENV PATH=/code/bin:$PATH
# If you want to develop, you'll need to comment this
# USER fluxuser
