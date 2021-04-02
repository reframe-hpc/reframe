FROM centos:7

# ReFrame user
RUN useradd -ms /bin/bash rfmuser

# ReFrame requirements
RUN \
  yum -y install gcc make git python3

# Install Tmod 3.2
RUN yum -y install environment-modules && \
    yum clean all && \
    rm -rf /var/cache/yum

USER rfmuser
