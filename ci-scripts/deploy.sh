#!/bin/bash

oldpwd=$(pwd)

usage()
{
    echo "Usage: $0 VERSION"
    echo "  Environment:"
    echo "    - GH_DEPLOY_CREDENTIALS=<user>:<token>"
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

if [ -z "$GH_DEPLOY_CREDENTIALS" ]; then
    _gh_creds_prefix=""
else
    _gh_creds_prefix="${GH_DEPLOY_CREDENTIALS}@"
fi


tmpdir=$(mktemp -d)
echo "Deploying ReFrame version $version ..."
echo "Working directory: $tmpdir ..."
cd $tmpdir
git clone --branch master https://${_gh_creds_prefix}github.com/reframe-hpc/reframe.git
cd reframe
./bootstrap.sh
found_version=$(./bin/reframe -V | sed -e 's/\(.*\)\+.*/\1/g')
if [ $found_version != $version ]; then
    echo "$0: version mismatch: found $found_version, but required $version" >&2
    exit 1
fi

./test_reframe.py
git tag -a v$version -m "ReFrame $version"
git push origin --tags

echo "Pushing of tags was successful!"
