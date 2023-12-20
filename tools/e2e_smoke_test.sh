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

# This script is used to test the system end-to-end on a single host running
# the `ubuntu_circuit_training`` docker container. It takes about ~30 minutes
# with 32vCPUs with no GPU. The script does **not** stop the background
# processes on exit because it is assumed to be running in a docker container.
#
# The script starts some background processes and will error out if they fail.
# The training process is logged to the console. The rest of the services are
# piped to files.
#
# If the training process hangs, the script will run forever. The most likely
# causes are 1) Reverb never starting but not having an error. 2) The collect
# jobs not collecting, but this is likely visible in the console by seeing
# "Filling up shuffle buffer (this may take a while): xxx of yyy".
set -e

# Flags
ROOT_DIR=./logs/run_00
SCRIPT_LOGS=""
REVERB_PORT=8008
REVERB_SERVER_IP=127.0.0.1
NETLIST_FILE=./circuit_training/environment/test_data/ariane/netlist.pb.txt
INIT_PLACEMENT=./circuit_training/environment/test_data/ariane/initial.plc
NUM_COLLECT_JOBS=4
USE_GPU=False

# Using keras-2
export TF_USE_LEGACY_KERAS=1

# Internal variables.
TIME_WAITING=0
SLEEP_TIME=60
HANDLER_EXIT_CODE=8

if [[ $# -lt 1 ]] ; then
  echo "Usage:"
  echo "--root_dir            [Where event logs and checkpoints will be stored]"
  echo "--reverb_port         [Port to use for reverb]"
  echo "--netlist_file        [Path to the netlist]"
  echo "--init_place          [Path to the initial placement file, e.g. *.plc]"
  echo "--num_collect_jobs    [Number of collect jobs to start]"
  echo "--script_logs         [Script logs, defaults to --root_dir value]"
  echo "--use_gpu             [Set to `True` to use the GPUs]"
  exit 1
fi

while [[ $# -gt -0 ]]; do
  key="$1"
  echo $key
  echo $2
  case $key in
      --root_dir)
      ROOT_DIR="$2" # Where to store event logs and checkpoints.
      shift
      ;;
      --script_logs)
      SCRIPT_LOGS="$2" # Where to store local logs.
      shift
      ;;
    --reverb_port)
      REVERB_PORT="$2"  # Port to start Reverb on locally.
      shift
      ;;
    --netlist_file)
      NETLIST_FILE="$2"  # Path to the netlist file.
      shift
      ;;
    --init_place)
      INIT_PLACEMENT="$2"  # Path to the initial placement file.
      shift
      ;;
    --num_collect_jobs)
      NUM_COLLECT_JOBS="$2"  # Number of collect jobs to start.
      shift
      ;;
    --use_gpu)
      USE_GPU="$2"  # Use GPU for training if set to `True`
      shift
      ;;
    *)
      echo "Unknown flag: $key"
      exit
      ;;
  esac
  shift # past argument or value
done

if [ -z "$SCRIPT_LOGS" ]; then
  echo "FYI: Local logs (--script_logs) cannot write to gcs. It is just a pipe."
  SCRIPT_LOGS=$ROOT_DIR
fi

handler() {
  echo "Caught interrupt signal (likely reverb). Check logs directory ${SCRIPT_LOGS}."
  echo "Exiting with code ${HANDLER_EXIT_CODE}."
  exit $HANDLER_EXIT_CODE
}

usr1_handler() {
  echo "Collect job failed (SIGUSR1). Check ${SCRIPT_LOGS}/collect_*.log."
  echo "Exiting with code ${HANDLER_EXIT_CODE}."
  exit $HANDLER_EXIT_CODE
}

usr2_handler() {
  echo "Train job failed (SIGUSR2). Exiting with code ${HANDLER_EXIT_CODE}."
  exit $HANDLER_EXIT_CODE
}

trap handler INT
trap usr1_handler USR1
trap usr2_handler USR2

start_background() {
  local -ir pid="$1"
  shift
  CUDA_VISIBLE_DEVICES=-1 "$@" || kill -INT -- "$pid"
}

start_background_collect() {
  local -ir pid="$1"
  shift
  CUDA_VISIBLE_DEVICES=-1 "$@" || kill -SIGUSR1 -- "$pid"
}

start_background_train() {
  local -ir pid="$1"
  shift
  "$@" || kill -SIGUSR2 -- "$pid"
}

REVERB_SERVER="${REVERB_SERVER_IP}:${REVERB_PORT}"
echo "Reverb server set to $REVERB_SERVER"

echo "Starting Reveb Server in the background."
start_background "$$" python3.9 -m circuit_training.learning.ppo_reverb_server  \
  --root_dir=${ROOT_DIR}  --port=${REVERB_PORT} &> ${SCRIPT_LOGS}/reverb.log &
echo "Logging reverb job ${i} to ${SCRIPT_LOGS}/reverb.log."

echo "Starting $NUM_COLLECT_JOBS collect jobs."
for i in $(eval echo "{1..$NUM_COLLECT_JOBS}")
do
 echo "Start collect job $i in the background..."
 start_background_collect "$$" python3.9 -m circuit_training.learning.ppo_collect \
  --root_dir=${ROOT_DIR} \
  --std_cell_placer_mode=dreamplace \
  --replay_buffer_server_address=${REVERB_SERVER} \
  --variable_container_server_address=${REVERB_SERVER} \
  --task_id=0 \
  --netlist_file=${NETLIST_FILE} \
  --init_placement=${INIT_PLACEMENT} &> ${SCRIPT_LOGS}/collect_${i}.log &
  echo "Logging collect job ${i} to ${SCRIPT_LOGS}/collect_${i}.log."
done

echo "Start Training job in the background but logging to console."
start_background_train "$$" python3.9 -m circuit_training.learning.train_ppo \
  --root_dir=${ROOT_DIR} \
  --replay_buffer_server_address=${REVERB_SERVER} \
  --variable_container_server_address=${REVERB_SERVER} \
  --std_cell_placer_mode=dreamplace \
  --gin_bindings='train.per_replica_batch_size=5' \
  --gin_bindings='train.num_iterations=1' \
  --gin_bindings='train.num_episodes_per_iteration=5' \
  --gin_bindings='train.num_epochs=4' \
  --netlist_file=${NETLIST_FILE} \
  --init_placement=${INIT_PLACEMENT} \
  --use_gpu=${USE_GPU} &

TRAIN_PROCESS=$!
# Wait for an error or until the training job stops.
while kill -0 $TRAIN_PROCESS &>/dev/null;
do
  WAITING_MINUTES=$((${TIME_WAITING} / 60))
  echo "It has been ~${WAITING_MINUTES}m. Sleeping ${SLEEP_TIME}s waiting for error or end."
  sleep $SLEEP_TIME
  TIME_WAITING=$((${TIME_WAITING} + ${SLEEP_TIME}))
done
