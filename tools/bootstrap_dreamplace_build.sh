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
# Creates the docker containers needed to build DREAMPlace and then builds
# DREAMPlace for multiple versions of Python.
set -x;
set -e;

# Builds circuit training core and then the dreamplace build container.
docker build --no-cache --tag circuit_training:ci -f docker/ubuntu_ci .
docker build --no-cache --tag circuit_training:dreamplace_build -f docker/ubuntu_dreamplace_build .

# Clones DREAMPlace from head.
if [ ! -d ../../DREAMPlace ] ; then
    git -C ../../  clone --recursive --branch circuit_training https://github.com/esonghori/DREAMPlace.git
    git -C ../../DREAMPlace/thirdparty/pybind11 pull https://github.com/pybind/pybind11.git
fi

# Force DreamPlace to use pybind11 2.10.3 which supports python 3.11.
git -C ../../DREAMPlace/thirdparty/pybind11 checkout v2.10.3

# Starts the container used for the build mounting DREAMPlace git,
# circuit_training /tools and then starting the build.
for python_version in python3.9 python3.10 python3.11
do
  docker run -it -v `cd ../../; pwd`/DREAMPlace:/dreamplace \
    -v $(pwd):/workspace circuit_training:dreamplace_build \
    bash /workspace/build_dreamplace.sh $python_version
done