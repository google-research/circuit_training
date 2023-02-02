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

# Builds circuit training core and then the dreamplace build container.
docker build --no-cache --tag circuit_training:ci -f docker/ubuntu_ci .
docker build --no-cache --tag circuit_training:dreamplace_build -f docker/ubuntu_dreamplace_build .

# Clones DREAMPlace from head.
git -C ../../  clone --recursive https://github.com/limbo018/DREAMPlace.git

# Starts the container used for the build mounting DREAMPlace git,
# circuit_training /tools and then starting the build.
for python_version in python3.7 python3.8 python3.9 python3.10
do
  docker run -it -v `cd ../../; pwd`/DREAMPlace:/dreamplace \
    -v $(pwd):/workspace circuit_training:dreamplace_build \
    bash /workspace/build_dreamplace.sh $python_version
done
