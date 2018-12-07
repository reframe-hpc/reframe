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
if [ $py_minor_version -ne 5 ]; then
    echo "$0: deployment script requires Python 3.5" >&2
    exit 1
fi


tmpdir=$(mktemp -d)
echo "Deploying ReFrame version $version ..."
echo "Working directory: $tmpdir ..."
cd $tmpdir
git clone https://github.com/eth-cscs/reframe.git
cd reframe
found_version=$(./reframe.py -V)
if [ $found_version != $version ]; then
    echo "$0: version mismatch: found $found_version, but required $version" >&2
    exit 1
fi

python3 -m venv venv.docs
source venv.docs/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -r docs/requirements.txt
./test_reframe.py
git tag -a v$version -m "ReFrame $version"
git push origin --tags
make -C docs
cd ..
git clone -b gh-pages https://github.com/eth-cscs/reframe.git reframe-doc
cd reframe-doc
rsync -avz ../reframe/docs/html/ .
echo "Please visit http://localhost:8000/ and " \
     "check that the documentation is fine."
echo "Press Ctrl-C to continue when ready."
python3 -m http.server
git commit -a -m "ReFrame $version documentation"
git push origin gh-pages
deactivate
cd $oldpwd
echo "Deployment was successful!"
