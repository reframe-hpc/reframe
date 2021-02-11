#!/bin/bash

requirements=(`cat requirements.txt | awk -F'=' '{print $1}' | grep -v \# | grep -v "coverage" | grep -v "pytest" | grep -v "pytest-forked" | grep -v "pytest-parlallel"| grep -v "setuptools"`)

### Get ReFrame version
version=`./bin/reframe --version | awk '{print $1}'`

### Download tarball to compute sha256
curl -O https://github.com/eth-cscs/reframe/archive/v${version}.tar.gz -o v${version}.tar.gz

shasum=`which sha256sum`
if [ $? -eq 0 ]; then
    hash=`${shasum} v${version}.tar.gz | awk '{print $1}'`
fi

shasum=`which shasum`
if [ $? -eq 0 ]; then
    hash=`${shasum} -a 256 v${version}.tar.gz | awk '{print $1}'`
fi


curl -O https://raw.githubusercontent.com/spack/spack/develop/var/spack/repos/builtin/packages/reframe/package.py -o package.py

# Insert version
sed -i -e "/branch='master')/a\ 
    version('${version}',       sha256='${hash}')
      " package.py

# fix spacing of the version entry
sed -i -e "s/version('${version}/    version('${version}/" package.py

## Remove runtime dependencies
sed -i -Ee "/depends_on.*py-.*',[[:blank:]]*type=.*run.*/D" package.py

## Insert runtime dependencies
for dep in ${requirements[@]}; do
    sed -i -e "/depends_on('py-setuptools.*)/a\ 
        depends_on('py-${dep}', type='run')
        " package.py

    # fix spacing of the depends_on entry
    sed -i -e "s/depends_on('py-${dep}', type='run')/    depends_on('py-${dep}', type='run')/" package.py
done

