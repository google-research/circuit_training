# Tools for Circuit Training

This directory contains tools and docker build scripts used for running
circuit training as well as executing unit and end-to-end tests.

## End-to-end smoke test

`e2e_smoke_test.sh` Executes a short (10-30+ minute) test that starts up a
reverb server, a number of collect jobs, and trains Ariane RISC-V for a short
period of time. The usage is as follows:

```shell
$ git clone https://github.com/google-research/circuit_training.git

$ export REPO_ROOT=$(pwd)/circuit_training
$ export TF_AGENTS_PIP_VERSION=tf-agents[reverb]
$ mkdir -p ${REPO_ROOT}/logs

# Build docker with tf-agents release defined by `TF_AGENTS_PIP_VERSION`.
$ docker build --pull --no-cache --tag circuit_training:core \
    --build-arg tf_agents_version="${TF_AGENTS_PIP_VERSION}" \
    -f "${REPO_ROOT}"/tools/docker/ubuntu_circuit_training ${REPO_ROOT}/tools/docker/

# Executes the end to end test.
$ docker run --rm -v ${REPO_ROOT}:/workspace --workdir /workspace circuit_training:core \
    bash tools/e2e_smoke_test.sh --root_dir /workspace/logs

# Some additional variations that can be run to test access google storage
# Writes the checkpoints and event logs to a google storage bucket.
$ docker run --rm -v ${REPO_ROOT}:/workspace --workdir /workspace circuit_training:core \
    bash tools/e2e_smoke_test.sh --script_logs /workspace/logs \
    --root_dir gs://<your bucket>/logs

# Writes to checkpionts to a storage buckets and reads the netlist from a bucket
# as well.
$ docker run --rm -v ${REPO_ROOT}:/workspace --workdir /workspace circuit_training:core \
    bash tools/e2e_smoke_test.sh --script_logs /workspace/logs \
    --root_dir gs://<your bucket>/logs \
    --netlist_file gs://<your bucket>/netlists/ariane/netlist.pb.txt \
    --init_place gs://<your bucket>/netlists/ariane/legalized.plc

```

## Docker build scripts.

There are a few docker build scripts. Instructions on how to use them are
in the scripts.

* ubuntu_ci: Used for presubmit testing along with `tox`.
* ubuntu_circuit_training: Suggested for using when running circuit training.
* ubuntu_dreamplace_build: Used to build binaries of DREAMPlace.


