#!/bin/bash

prefix=${1:-html}

if [[ x$OSTYPE =~ x"darwin" ]]; then
    symlink="ln -sfh"
else
    symlink="ln -sfn"
fi

link_old_docs()
{
    echo "linking old docs for $1 ..."
    cd $1 && $symlink ../_old _old && cd - > /dev/null
}


link_old_docs $prefix/master

for d in $prefix/v*; do
    # Verify that $d is actually a version
    if ! echo $(basename $d) | grep -e 'v[0-9]\+\.[0-9]\+' > /dev/null; then
        echo Skipping non version directory $d ...
        continue
    fi

    link_old_docs $d
done
