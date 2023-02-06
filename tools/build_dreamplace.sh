# coding=utf-8
# Copyright 2021 The Circuit Training Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#!/bin/bash
# Builds DREAMPlace and expects to be called from bootstrap_dreamplace_build.sh
# and executed within a docker container with all DREAMPlace required libs.
#
# Example: bash build_dreamplace.sh python3.9
#
set -x;
set -e;

# Builds DREAMPlace.
mkdir -p /dreamplace/build
cd /dreamplace/build
cmake .. -DCMAKE_INSTALL_PREFIX=dreamplace_build -DPYTHON_EXECUTABLE=$(which $1)
make -j10
make install

# Packages (tar/zip) DREAMPlace to make it ready to be uploaded.
git config --global --add safe.directory /dreamplace
GIT_HASH=$(git rev-parse --short HEAD)
printf -v BUILD_DATE '%(%Y%m%d)T' -1
BINARY_NAME="dreamplace_${BUILD_DATE}_${GIT_HASH}_${1}"

echo $BINARY_NAME

tar -cvzf ${BINARY_NAME}.tar.gz -C dreamplace_build .
chmod 744 ${BINARY_NAME}.tar.gz
cp ${BINARY_NAME}.tar.gz /workspace/

# Cleans the environment by deleting the build directory.
rm -rf /dreamplace/build
