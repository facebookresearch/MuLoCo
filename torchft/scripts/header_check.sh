#!/bin/bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -e

FILES="$(rg -F 'Meta Platforms' --files-without-match --type rust --type python || true)"

# exit 1 if there are files without the match and print them.
if [ -n "$FILES" ]; then
  echo "Found files without header:"

  for file in $FILES;
  do
    echo $file
  done

  exit 1
fi

exit 0
