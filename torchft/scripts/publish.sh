#!/bin/bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -ex

docker build -t torchft .
docker run --rm -v $(pwd):/io torchft build --release --interpreter python3.12
docker run --rm -v $(pwd):/io torchft build --release --interpreter python3.11
docker run --rm -v $(pwd):/io torchft build --release --interpreter python3.10
docker run --rm -v $(pwd):/io torchft build --release --interpreter python3.9
docker run --rm -v $(pwd):/io torchft build --release --interpreter python3.8
python3 -m twine upload target/wheels/*manylinux2014*
