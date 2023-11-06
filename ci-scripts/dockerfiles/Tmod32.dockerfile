FROM centos:7

# Install Tmod 3.2.10
RUN yum -y install environment-modules && \
    yum clean all && \
    rm -rf /var/cache/yum
