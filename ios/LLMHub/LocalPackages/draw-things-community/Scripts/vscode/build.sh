#!/usr/bin/env bash

set -euo pipefail

if [ $# -eq 0 ] ; then
    WORKSPACE=$(git rev-parse --show-toplevel)
else
    WORKSPACE=$1
fi

bazel run //Scripts/compdb:compdb -- ${WORKSPACE}
TARGETS=`bazel query "kind(swift_binary,//...)"`
bazel build --compilation_mode=dbg ${TARGETS}
mkdir -p ${WORKSPACE}/.index/store && mkdir -p ${WORKSPACE}/.index/db
find ${WORKSPACE}/bazel-out/k8-dbg/bin -name "*.indexstore" | xargs -I {} rsync -a {}/v5 ${WORKSPACE}/.index/store
