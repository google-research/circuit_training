# Circuit training at scale with Ariane RISC-V

This document details how to scale Circuit Training using Google Cloud as the
example and a block from Ariane RISC-V as the target. These instructions are
modified over time to illustrate how to scale. The numbers from our training run
are in the [history section](#ariane-risc-v-jan-2022).

## Running at Scale

In order to run at scale, a large number of collect jobs are needed. It is not
unreasonable to have 500+ collect jobs to drive enough episodes to keep 8 NVIDIA
V100s busy.

For the Ariane RISC-V example we used the configuration in the list below. The
intention is the information will allow you to run an experiment at scale
using your orchestration system of choice. Our assumption is that you would be
using your own orchestration, e.g. SLURM, Kubernetes, or similar product.
Although simple, we flushed out some issue using `bash` and remotely executing
scripts via `gcloud compute ssh`. If you are having issues, please
[open an issue](https://github.com/google-research/circuit_training/issues).
We want you to be successful.

For the training we utilized the following servers and jobs:

*   1 Replay Buffer(Reverb)/Eval server 32vCPUs (n1-standard-32)
    *   1 Replay Buffer(Reverb) job
    *   1 Eval job
*   20 Collect servers 96vCPUs (n1-standard-96)
    *   Each server running 25 collect jobs for a total of 500.
*   1 Training server: 8xV100s (n1-standard-96)
    *   1 Training job

Each iteration of training clears the replay buffer due to the on-policy nature
of the PPO algorithm. This results in a large amount of training time spent
waiting for the replay buffer to refill based on the last policy. 500 collect
agents worked well for this example to feed 8xV100s. It is not necessary to have
that many collect jobs. Having less will slow down total training time
(walltime) but will not impact the quality of the result. We have noticed that
using a smaller `per_replica_batch_size` or smaller `num_episodes_per_iteration`
can reduce quality.

### Execution outline and highlights

The steps below are not intended to be step-by-step. Listed below are the
commands we used to start each of the job types and the server type we used. We
have left out how we orchestrated starting 500 collect jobs across 20 servers
and other details that we felt would be noise given your environment and
orchestration tools will be different.

#### Create the servers

##### Common ENV Vars

```shell
$ export PROJECT=<Your Google Cloud Project>
$ export ZONE=<Zone you are using>
$ export TRAIN_INSTANCE_NAME=ct-train
$ export REVERB_INSTANCE_NAME=ct-reverb
```

##### 20 Collect servers

Create 20 96vCPU servers to host the ~500 collect jobs. Training will work with
less collect jobs and get the same result; but each iteration of training is
gated on collect as the replay buffer is cleared after each iteration.

```shell
# This needs to be run 20 times with a different `$INSTANCE_NAME` each time.
$ export INSTANCE_NAME=ct-collect-00
$ gcloud compute instances create ${INSTANCE_NAME} \
    --project=${PROJECT} \
    --zone=${ZONE} \
    --machine-type=n1-standard-96 \
    --maintenance-policy=TERMINATE \
    --image-family=tf-ent-latest-cpu \
    --image-project=deeplearning-platform-release \
    --boot-disk-size=100GB \
    --boot-disk-type=pd-ssd \
    --scopes="https://www.googleapis.com/auth/cloud-platform"

```

##### 1 Replay Buffer (Reverb)/Eval server

Create one 32vCPU server that will host the reverb and eval job.

```shell
$ gcloud compute instances create ${REVERB_INSTANCE_NAME} \
    --project=${PROJECT} \
    --zone=${ZONE} \
    --machine-type=n1-standard-32 \
    --maintenance-policy=TERMINATE \
    --image-family=tf-ent-latest-cpu \
    --image-project=deeplearning-platform-release \
    --boot-disk-size=100GB \
    --boot-disk-type=pd-ssd \
    --scopes="https://www.googleapis.com/auth/cloud-platform"

```

##### 1 Training server

The training server is configured with 8x NVIDIA V100s.

```shell
$ gcloud compute instances create ${TRAIN_INSTANCE_NAME} \
    --project=${PROJECT} \
    --zone=${ZONE} \
    --machine-type=n1-standard-96 \
    --accelerator=count=8,type=nvidia-tesla-v100 \
    --maintenance-policy=TERMINATE \
    --image-family=tf-ent-latest-gpu \
    --image-project=deeplearning-platform-release \
    --boot-disk-size=100GB \
    --boot-disk-type=pd-ssd \
    --scopes="https://www.googleapis.com/auth/cloud-platform"
```

#### Copy code and set environment variables.

While this can be run at HEAD with nightly builds from the upstream libraries,
we suggest using a branched version. The instructions below setup the docker
images with the latest branch. If you want to try head, these
[instructions](https://github.com/google-research/circuit_training/blob/main/README.md#head)
provide the breadcrumbs needed to build the images with head/nightly.

Each host server is going to need:

##### Access to the circuit training code

Our approach was to copy the code to each server. The instructions below assume
the circuit training repo root is located at `$HOME/run_ct/circuit_training` on
the host systems.

```shell
$ export CT_VERSION=0.0.4
$ git clone https://github.com/google-research/circuit_training.git
$ git -C $(pwd)/circuit_training checkout r${CT_VERSION}
```

##### A service account key

A
[service account key](https://cloud.google.com/iam/docs/creating-managing-service-account-keys)
with access to write to a
[Google Cloud Storage Bucket](https://cloud.google.com/storage/docs/creating-buckets).

##### Environment variables

How this is done will be based on your orchestration system. These are the
environment variable we made available on each of the host machines.

```shell
$ export REPO_ROOT=$HOME/run_ct/circuit_training
## These first variables are only needed for building the docker images.
# Version of Circuit Training to use, which is also used to build images.
$ export TF_AGENTS_PIP_VERSION=tf-agents[reverb]
$ export CT_VERSION=0.0.3
# The docker is python3.9 only.
$ export PYTHON_VERSION=python3.9
$ export DREAMPLACE_PATTERN=dreamplace_20231214_c5a83e5_${PYTHON_VERSION}.tar.gz
## These variables are used at runtime and passed to the docker container.
$ export REVERB_PORT=8008
$ export REVERB_SERVER="<IP Address of the Reverb Server>:${REVERB_PORT}"
$ export ROOT_DIR=<Path to network storage, e.g. gs://my-bucket/logs/run_00>
$ export NETLIST_FILE=./circuit_training/environment/test_data/ariane/netlist.pb.txt
$ export INIT_PLACEMENT=./circuit_training/environment/test_data/ariane/initial.plc
$ export GLOBAL_SEED=333

```

We have also uploaded a toy netlist that has a couple macros and can easily be
trained on CPU in a few minutes. It is great for smoke testing.

```shell
# Use these vars to use the toy netlist, otherwise skip these commands.
$ export NETLIST_FILE=gs://rl-infra-public/circuit-training/netlist/toy/netlist.pb.txt
$ export INIT_PLACEMENT=gs://rl-infra-public/circuit-training/netlist/toy/initial.plc
# If using the toy example make sure on the training job to set 
# --sequence_length=3, or it will hang looking for episodes.

```

#### Docker

We suggest using Docker as we use the same images for continuous integration
testing and they include the required dependencies. You can either store a
docker images in a central location or create the image on each of the servers.

```shell
# For Collect and Reverb/Eval Jobs
$ docker build --pull --no-cache --tag circuit_training:core \
    --build-arg tf_agents_version="${TF_AGENTS_PIP_VERSION}" \
    --build-arg dreamplace_version="${DREAMPLACE_PATTERN}" \
    --build-arg placement_cost_binary="plc_wrapper_main_${CT_VERSION}" \
    -f "${REPO_ROOT}"/tools/docker/ubuntu_circuit_training ${REPO_ROOT}/tools/docker/

# For training job with NVIDIA GPU support.
$ docker build --pull --no-cache --tag circuit_training:core \
    --build-arg base_image=nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04 \
    --build-arg tf_agents_version="${TF_AGENTS_PIP_VERSION}" \
    --build-arg dreamplace_version="${DREAMPLACE_PATTERN}" \
    --build-arg placement_cost_binary="plc_wrapper_main_${CT_VERSION}" \
    -f "${REPO_ROOT}"/tools/docker/ubuntu_circuit_training ${REPO_ROOT}/tools/docker/
```

#### Job execution

This section outlines the jobs that will need to be started. The Docker
containers are started detached (`-d`). To verify they are running attach to the
container with `docker attach` and exit with `Ctrl-P` followed by `Ctrl-Q`.

All of the commands assume a
[service account key](https://cloud.google.com/iam/docs/creating-managing-service-account-keys)
with access to write to the
[Google Cloud Storage Bucket](https://cloud.google.com/storage/docs/creating-buckets)
has been copied into the root of the cloned github repo. Each command is to be
executed from the root of the cloned repo.

The order that the jobs are started should not matter. Each job waits for what
it needs from the other jobs before moving forward. If the collect jobs stop for
some reason the train job will keep training on stale data.

###### Replay Buffer job

Start the reverb job on the Replay Buffer(Reverb)/Eval server. Make sure to set
`-p` to the port reverb will be running on. The reverb job will continuously
write to console

```
Waiting for `wait_predicate_fn`. Block execution. Sleeping for 1 seconds.
```

until the training job is started and writes out the initial collect policy.

```shell
$ docker run --rm -d -it -p 8008:8008 -e "GOOGLE_APPLICATION_CREDENTIALS=/workspace/cloud_key.json" \
     --rm -it -v ${REPO_ROOT}:/workspace -w /workspace/ circuit_training:core  \
     python3.9 -m circuit_training.learning.ppo_reverb_server \
       --global_seed=${GLOBAL_SEED} \
       --root_dir=${ROOT_DIR}  \
       --port=${REVERB_PORT}
```

###### Train job

Start the training job on the training server. Remember to use the Docker image
created with GPU support. 200 iterations will be about 100K steps.

```shell
$ docker run --network host -d -e "GOOGLE_APPLICATION_CREDENTIALS=/workspace/cloud_key.json" \
     --gpus all  --rm -it -v ${REPO_ROOT}:/workspace -w /workspace/ circuit_training:core  \
     python3.9 -m circuit_training.learning.train_ppo \
       --root_dir=${ROOT_DIR} \
       --std_cell_placer_mode=dreamplace \
       --replay_buffer_server_address=${REVERB_SERVER} \
       --variable_container_server_address=${REVERB_SERVER} \
       --sequence_length=134 \
       --gin_bindings='train.num_iterations=200'\
       --netlist_file=${NETLIST_FILE} \
       --init_placement=${INIT_PLACEMENT} \
       --global_seed=${GLOBAL_SEED} \
       --use_gpu

# If using the toy netlist, some args need changed. Use this command instead.
$ docker run --network host -d -e "GOOGLE_APPLICATION_CREDENTIALS=/workspace/cloud_key.json" \
     --gpus all  --rm -it -v ${REPO_ROOT}:/workspace -w /workspace/ circuit_training:core  \
     python3.9 -m circuit_training.learning.train_ppo \
       --root_dir=${ROOT_DIR} \
       --std_cell_placer_mode=dreamplace \
       --replay_buffer_server_address=${REVERB_SERVER} \
       --variable_container_server_address=${REVERB_SERVER} \
       --sequence_length=3 \
       --gin_bindings='train.num_iterations=200' \
       --gin_bindings='train.num_episodes_per_iteration=32' \
       --gin_bindings='train.per_replica_batch_size=64' \
       --gin_bindings='CircuittrainingPPOLearner.summary_interval=12' \
       --gin_bindings='CircuitPPOAgent.debug_summaries=True' \
       --netlist_file=${NETLIST_FILE} \
       --init_placement=${INIT_PLACEMENT} \
       --global_seed=${GLOBAL_SEED} \
       --use_gpu

```

###### Collect jobs

Each of the 20 collect servers should run 25 collect jobs. While you would not
orchestrate it this way, the bash script below illustrates the point. The
following command would need to be run on each of the 20 collect servers.
Watching the CPU usage is the best way to figure out the max number of collect
jobs your server type can handle.

```bash
for i in $(seq 1 23); do
  docker run --network host -d -e "GOOGLE_APPLICATION_CREDENTIALS=/workspace/cloud_key.json" \
  --rm -it -v ${REPO_ROOT}/circuit_training:/workspace -w /workspace/ circuit_training:core  \
     python3.9 -m circuit_training.learning.ppo_collect \
  --root_dir=${ROOT_DIR} \
  --std_cell_placer_mode=dreamplace \
  --replay_buffer_server_address=${REVERB_SERVER} \
  --variable_container_server_address=${REVERB_SERVER} \
  --task_id=${i} \
  --netlist_file=${NETLIST_FILE} \
  --init_placement=${INIT_PLACEMENT} \
  --global_seed=${GLOBAL_SEED} \
  --logtostderr
```

It is ok for the `task_id` to repeat across or even on a given server. To get
logging of a single collect job in Tensorboard set a single collect job to have
a `task_id=0`.

###### Eval job

Start the eval job on the replay buffer(Reverb)/eval server. Eval could be run
on a seperate server with the same command below if desired.

```shell
$ docker run --network host -d -e "GOOGLE_APPLICATION_CREDENTIALS=/workspace/cloud_key.json" \
     --rm -it -v $(pwd):/workspace -w /workspace/ circuit_training:core  \
     python3.9 -m circuit_training.learning.eval \
       --root_dir=${ROOT_DIR} \
       --std_cell_placer_mode=dreamplace \
       --variable_container_server_address=${REVERB_SERVER} \
       --netlist_file=${NETLIST_FILE} \
       --init_placement=${INIT_PLACEMENT} \
       --global_seed=${GLOBAL_SEED} \
       --output_placement_save_dir=./
```

#### Monitoring

From the replay buffer(Reverb)/eval server or a local workstation run
Tensorboard to monitor the results.

```shell
$ tensorboard dev upload --logdir $ROOT_DIR
```

## Ariane RISC-V Jan 2022

The numbers below are from a run we did in January of 2022 on Google Cloud. Much
has changed and we are not currently taking the time to redo the experiment. The
instructions on how to scale Circuit Training are evolving and this section
exists to preserve the results from Jan 2022. 

### Results

The results in the table below are reported for training from scratch. We
trained with 3 different seeds run 3 times each. This is slightly different than
what was used in the paper (8 runs each with a different seed), but better
captures the different sources of variability. Our results training from scratch
are comparable or better than the reported results in the
[paper](https://www.nature.com/articles/s41586-021-03544-w.epdf?sharing_token=tYaxh2mR5EozfsSL0WHZLdRgN0jAjWel9jnR3ZoTv0PW0K0NmVrRsFPaMa9Y5We9O4Hqf_liatg-lvhiVcYpHL_YQpqkurA31sxqtmA-E1yNUWVMMVSBxWSp7ZFFIWawYQYnEXoBE4esRDSWqubhDFWUPyI5wK_5B_YIO-D_kS8%3D)
(on page 22) which used fine-tuning from a pre-trained model. We are training
from scratch because we cannot publish the pre-trained model at this time and
the released code can provide comparable results. The metrics listed in the
table correspond to the output optimal placement. Training was run for 200
iterations which results in ~107K steps (gradient updates). This
[tensorboard link](https://tensorboard.dev/experiment/NRlmrDeOT2i4QV334hrywQ)
contains the raw results of the runs for full transparency and usefulness.

Run id | Seed | Proxy Wirelength | Proxy Congestion | Proxy Density | Step
------ | ---- | ---------------- | ---------------- | ------------- | -------
run_00 | 111  | 0.1051           | 0.8746           | 0.5154        | 77,184
run_01 | 111  | 0.0979           | 0.8749           | 0.5217        | 51,992
run_02 | 111  | 0.1052           | 0.9069           | 0.5289        | 94,872
run_03 | 222  | 0.1021           | 0.9667           | 0.5799        | 33,232
run_04 | 222  | 0.0963           | 0.9945           | 0.6795        | 103,984
run_05 | 222  | 0.1060           | 1.0352           | 0.5886        | 37,520
run_06 | 333  | 0.1012           | 0.8738           | 0.5110        | 46,096
run_07 | 333  | 0.0977           | 0.8684           | 0.5109        | 35,912
run_08 | 333  | 0.1004           | 0.8613           | 0.5160        | 48,776

_        | Proxy Wirelength | Proxy Congestion | Proxy Density
-------- | ---------------- | ---------------- | -------------
**mean** | 0.1013           | 0.9174           | 0.5502
**std**  | 0.0036           | 0.0647           | 0.0568

Applying coordinated descent after training resulted in improved proxy numbers
for complex blocks like those used in TPUs as referenced in the paper. However,
for the simpler Ariane RISC-V there were modest (1-2%) improvements to proxy
wirelength and congestion.

_        | Proxy Wirelength | Proxy Congestion | Proxy Density
-------- | ---------------- | ---------------- | -------------
**mean** | 0.0988           | 0.9077           | 0.5513
**std**  | 0.0053           | 0.0621           | 0.0589

### Hyperparameters

Below are the hyperparameters and the values that were used for our experiments.
Some hyperparameters are changed from the paper to make the training more stable
for the Ariane block. The hyperparameters listed in the paper were set for TPU
blocks which have different characteristics. For training, we use the clipping
version of proximal policy optimization (PPO)
([Schulman et al., 2017](https://arxiv.org/abs/1707.06347)) without the KL
divergence penalty implemented by
[tf-agents](https://www.tensorflow.org/agents). The default for the training
hyperparameters, if not specified in the table, is the same as the defaults in
the tf-agents.

| Configuration              | Default Value     | Comments                  |
| -------------------------- | ----------------- | ------------------------- |
| **Proxy reward             |                   |                           |
: calculation**              :                   :                           :
| wirelength_weight          | 1.0               |                           |
| density_weight             | 1.0               | Changed from 0.1 in the   |
:                            :                   : paper, since it produces  :
:                            :                   : more stable training from :
:                            :                   : scratch on Ariane blocks. :
| congestion_weight          | 0.5               | Changed from 0.1 in the   |
:                            :                   : paper, since it produces  :
:                            :                   : more stable training from :
:                            :                   : scratch on Ariane blocks. :
| **Standard cell            |                   |                           |
: placement**                :                   :                           :
| num_steps                  | [100, 100, 100]   |                           |
| io_factor                  | 1.0               |                           |
| move_distance_factors      | [1, 1, 1]         |                           |
| attract_factors            | [100, 1e-3, 1e-5] |                           |
| repel_factors              | [0, 1e6, 1e7]     |                           |
| **Environment              |                   |                           |
: observation**              :                   :                           :
| max_num_nodes              | 4700              |                           |
| max_num_edges              | 28400             |                           |
| max_grid_size              | 128               |                           |
| default_location_x         | 0.5               |                           |
| default_location_y         | 0.5               |                           |
| **Model architecture**     |                   |                           |
| num_gcn_layers             | 3                 |                           |
| edge_fc_layers             | 1                 |                           |
| gcn_node_dim               | 8                 |                           |
| dirichlet_alpha            | 0.1               |                           |
| policy_noise_weight        | 0.0               |                           |
| **Training**               |                   |                           |
| optimizer                  | Adam              |                           |
| learning_rate              | 4e-4              |                           |
| sequence_length            | 134               |                           |
| num_episodes_per_iteration | 1024              |                           |
| per_replica_batch_size     | 128               |                           |
| num_epochs                 | 4                 |                           |
| value_pred_loss_coef       | 0.5               |                           |
| entropy_regularization     | 0.01              |                           |
| importance_ratio_clipping  | 0.2               |                           |
| discount_factor            | 1.0               |                           |
| entropy_regularization     | 0.01              |                           |
| value_pred_loss_coef       | 0.5               |                           |
| gradient_clipping          | 1.0               |                           |
| use_gae                    | False             |                           |
| use_td_lambda_return       | False             |                           |
| log_prob_clipping          | 0.0               |                           |
| policy_l2_reg              | 0.0               |                           |
| value_function_l2_reg      | 0.0               |                           |
| shared_vars_l2_reg         | 0.0               |                           |
