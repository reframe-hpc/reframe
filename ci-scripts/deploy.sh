#!/bin/bash

oldpwd=$(pwd)

usage()
{
    echo "Usage: $0 VERSION"
}

_onerror()
{
    exitcode=$?
    echo "$0: ReFrame deployment failed!"
    echo "$0: command \`$BASH_COMMAND' failed (exit code: $exitcode)"
    cd $oldpwd
    exit $exitcode
}

trap _onerror ERR

version=$1
if [ -z $version ]; then
    echo "$0: missing version number" >&2
    usage
    exit 1
fi

py_minor_version=$(python3 -c 'import sys; print(sys.version_info.minor)')
if [ $py_minor_version -lt 5 ]; then
    echo "$0: deployment script requires Python>=3.5" >&2
    exit 1
fi

tmpdir=$(mktemp -d)
echo "Deploying ReFrame version $version ..."
echo "Working directory: $tmpdir ..."
cd $tmpdir
git clone https://github.com/vkarak/reframe.git
cd reframe
git checkout feat/immediate-install-ci-script
./bootstrap.sh
found_version=$(./bin/reframe -V | sed -e 's/ (.*)//g')
if [ $found_version != $version ]; then
    echo "$0: version mismatch: found $found_version, but required $version" >&2
    exit 1
fi

./test_reframe.py
git tag -a v$version -m "ReFrame $version"
git push origin --tags

# We need this for running the setup.py of ReFrame
export PYTHONPATH=$(pwd)/external:$PYTHONPATH

# We create a virtual environment here just for the deployment
python3 -m venv venv.deployment
source venv.deployment/bin/activate
python3 -m pip install --upgrade pip setuptools wheel twine
python3 setup.py sdist bdist_wheel
python3 -m twine upload dist/*
deactivate
cd $oldpwd
echo "Deployment was successful!"
