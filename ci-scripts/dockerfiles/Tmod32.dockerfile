FROM centos:7

RUN sed -i s/mirror.centos.org/vault.centos.org/g /etc/yum.repos.d/*.repo && \
    sed -i s/^#.*baseurl=http/baseurl=http/g /etc/yum.repos.d/*.repo && \
    sed -i s/^mirrorlist=http/#mirrorlist=http/g /etc/yum.repos.d/*.repo

# Install Tmod 3.2.10
RUN yum -y install environment-modules && \
    yum clean all && \
    rm -rf /var/cache/yum
