#!/usr/bin/env bash

set -euo pipefail

GIT_CONFIG=$(git rev-parse --git-dir)
GIT_ROOT=$(git rev-parse --show-toplevel)

cd $GIT_ROOT

bazel build Apps/DrawThings --config=release --ios_multi_cpus=arm64
