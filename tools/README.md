# Tools for Circuit Training

This directory contains tools and docker build scripts used for running
circuit training as well as executing unit and end-to-end tests.

## End-to-end smoke test

`e2e_smoke_test.sh` Executes a short (10-30+ minute) test that starts up a
reverb server, a number of collect jobs, and trains Ariane RISC-V for a short
period of time. The usage is as follows:

### Common setup for both paths

```shell
$ export CT_VERSION=0.0.4
$ git clone https://github.com/google-research/circuit_training.git
$ git -C $(pwd)/circuit_training checkout r${CT_VERSION}

$ export REPO_ROOT=$(pwd)/circuit_training
$ export TF_AGENTS_PIP_VERSION=tf-agents[reverb]
$ export PYTHON_VERSION=python3.9
$ export DREAMPLACE_PATTERN=dreamplace_20231214_c5a83e5_${PYTHON_VERSION}.tar.gz
$ mkdir -p ${REPO_ROOT}/logs
```

#### CPU setup and execution

```shell
# Build docker with tf-agents release defined by `TF_AGENTS_PIP_VERSION`.
$ docker build --pull --no-cache --tag circuit_training:core \
    --build-arg tf_agents_version="${TF_AGENTS_PIP_VERSION}" \
    --build-arg dreamplace_version="${DREAMPLACE_PATTERN}" \
    --build-arg placement_cost_binary="plc_wrapper_main_${CT_VERSION}" \
    -f "${REPO_ROOT}"/tools/docker/ubuntu_circuit_training ${REPO_ROOT}/tools/docker/

# Executes the end to end test.
$ docker run --rm -v ${REPO_ROOT}:/workspace --workdir /workspace circuit_training:core \
    bash tools/e2e_smoke_test.sh --root_dir /workspace/logs

# Some additional variations to test accessing Google Cloud Storage
# Writes the checkpoints and event logs to a Google Storage bucket.
$ docker run --rm -v ${REPO_ROOT}:/workspace --workdir /workspace circuit_training:core \
    bash tools/e2e_smoke_test.sh --script_logs /workspace/logs \
    --root_dir gs://<your bucket>/logs

# Writes to checkpoints to a storage buckets and reads the netlist from a bucket
# as well.
$ docker run --rm -v ${REPO_ROOT}:/workspace --workdir /workspace circuit_training:core \
    bash tools/e2e_smoke_test.sh --script_logs /workspace/logs \
    --root_dir gs://<your bucket>/logs \
    --netlist_file gs://<your bucket>/netlists/ariane/netlist.pb.txt \
    --init_place gs://<your bucket>/netlists/ariane/legalized.plc
```


#### GPU setup and execution

```shell
# Build docker with tf-agents release defined by `TF_AGENTS_PIP_VERSION`.
$ docker build --pull --no-cache --tag circuit_training:core \
    --build-arg base_image=nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04 \
    --build-arg tf_agents_version="${TF_AGENTS_PIP_VERSION}" \
    --build-arg dreamplace_version="${DREAMPLACE_PATTERN}" \
    --build-arg placement_cost_binary="plc_wrapper_main_${CT_VERSION}" \
    -f "${REPO_ROOT}"/tools/docker/ubuntu_circuit_training ${REPO_ROOT}/tools/docker/

# Executes the end to end test.
$ docker run  --gpus all --rm -v ${REPO_ROOT}:/workspace --workdir /workspace circuit_training:core \
    bash tools/e2e_smoke_test.sh --root_dir /workspace/logs --use_gpu True
```

## Docker build scripts.

There are a few docker build scripts. Instructions on how to use them are
in the scripts.

* ubuntu_ci: Used for presubmit testing along with `tox`.
* ubuntu_circuit_training: Suggested for using when running circuit training.
* ubuntu_dreamplace_build: Used to build binaries of DREAMPlace.


