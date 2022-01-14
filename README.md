<!--* freshness: {
  owner: 'azalia'
  owner: 'tobyboyd'
  owner: 'sguada'
  owner: 'morpheus-oss-team'
  reviewed: '2022-01-11'
  review_interval: '12 months'
} *-->


# Circuit Training: An open-source framework for generating chip floor plans with distributed deep reinforcement learning.

*Circuit Training* is an open-source framework for generating chip floor plans
with distributed deep reinforcement learning. This framework reproduces the
methodology published in the Nature 2021 paper:

*[A graph placement methodology for fast chip design.](https://www.nature.com/articles/s41586-021-03544-w) Mirhoseini, A., Goldie, A.,
Yazgan, M., Jiang, J.W., Songhori, E., Wang, S., Lee, Y.J., Johnson, E., Pathak,
O., Nazi, A. and Pak, J., 2021. Nature, 594(7862), pp.207-212. [[PDF]](https://www.nature.com/articles/s41586-021-03544-w.epdf?sharing_token=tYaxh2mR5EozfsSL0WHZLdRgN0jAjWel9jnR3ZoTv0PW0K0NmVrRsFPaMa9Y5We9O4Hqf_liatg-lvhiVcYpHL_YQpqkurA31sxqtmA-E1yNUWVMMVSBxWSp7ZFFIWawYQYnEXoBE4esRDSWqubhDFWUPyI5wK_5B_YIO-D_kS8%3D)*


Our hope is that *Circuit Training* will foster further collaborations between
academia and industry, and enable advances in deep reinforcement learning for
Electronic Design Automation, as well as, general combinatorial and decision
making optimization problems. Capable of optimizing chip blocks with over
hundreds of macros, *Circuit Training* automatically generates floor plans in
hours, whereas baseline methods often require human experts in the loop and can
take months

Circuit training is built on top of [TF-Agents](https://github.com/tensorflow/agents)
and [TensorFlow 2.x](https://www.tensorflow.org/) with support for
eager execution, distributed training across multiple GPUs, and distributed data
collection scaling to 100s of actors.

## Table of contents
<a href='#Features'>Features</a><br>
<a href='#Installation'>Installation</a><br>
<a href='#QuickStart'>Quick start</a><br>
<a href='#Results'>Results</a><br>
<a href='#Testing'>Testing</a><br>
<a href='#Releases'>Releases</a><br>
<a href='#Contributing'>How to contribute</a><br>
<a href='#Principles'>AI Principles</a><br>
<a href='#Contributors'>Contributors</a><br>
<a href='#Citation'>How to cite</a><br>
<a href='#Disclaimer'>Disclaimer</a><br>

<a id='Features'></a>

## Features
* Places netlists with hundreds of macros and millions of stdcells (in clustered format).
* Computes both macro location and orientation (flipping).
* Optimizes multiple objectives including wirelength, congestion, and density.
* Supports blockages on the grid, to model clock strap or macro blockage.
* Supports macro-to-macro, macro-to-boundary spacing constraints.
* Allows users to specify their own technology parameters, e.g. and routing resources (in routes per micron) and macro routing allocation.
* **Coming soon**: Tools for generating a clustered netlist given a netlist in common formats (Bookshelf and LEF/DEF).
* **Coming soon**: Generates macro placement tcl command compatible with major EDA tools (Innovus, ICC2).


<a id='Installation'></a>
## Installation

Circuit Training requires:

   * Installing TF-Agents which includes Reverb and TensorFlow.
   * Downloading the placement cost binary into your system path.
   * Downloading the circuit-training code.

Using the code at `HEAD` with the nightly release of tf-agents is recommended.

```shell
# Installs TF-Agents with nightly versions of Reverb and TensorFlow 2.x
$  pip install tf-agents-nightly[reverb]
# Copies the placement cost binary to /usr/local/bin and makes it executable.
$  sudo curl https://storage.googleapis.com/rl-infra-public/circuit-training/placement_cost/plc_wrapper_main \
     -o  /usr/local/bin/plc_wrapper_main
$  sudo chmod 555 /usr/local/bin/plc_wrapper_main
# Clones the circuit-training repo.
$  git clone https://github.com/google-research/circuit-training.git
```

<a id='QuickStart'></a>
## Quick start

This quick start places the Ariane RISC-V CPU macros by training the deep
reinforcement policy from scratch. The `num_episodes_per_iteration` and
`global_batch_size` used below were picked to work on a single machine training
on CPU. The purpose is to illustrate a running system, not optimize the result.
The result of a few thousand steps is shown in this
[tensorboard](https://tensorboard.dev/experiment/rBEQZlV8T0mEokys3Pkj5g).
The full scale Ariane RISC-V experiment matching the paper is detailed in
[Circuit training for Ariane RISC-V](./docs/ARIANE.md).

The following jobs will be created by the steps below:

   * 1 Replay Buffer (Reverb) job
   * 1-3 Collect jobs
   * 1 Train job
   * 1 Eval job

Each job is started in a `tmux` session. To switch between sessions use
`ctrl + b` followed by `s` and then select the specified session.

```shell
# Sets the environment variables needed by each job. These variables are
# inherited by the tmux sessions created in the next step.
$  export ROOT_DIR=./logs/run_00
$  export REVERB_PORT=8008
$  export REVERB_SERVER="127.0.0.1:${REVERB_PORT}"
$  export NETLIST_FILE=./circuit_training/environment/test_data/ariane/netlist.pb.txt
$  export INIT_PLACEMENT=./circuit_training/environment/test_data/ariane/initial.plc

# Creates all the tmux sessions that will be used.
$  tmux new-session -d -s reverb_server && \
   tmux new-session -d -s collect_job_00 && \
   tmux new-session -d -s collect_job_01 && \
   tmux new-session -d -s collect_job_02 && \
   tmux new-session -d -s train_job && \
   tmux new-session -d -s eval_job && \
   tmux new-session -d -s tb_job

# Starts the Replay Buffer (Reverb) Job
$  tmux attach -t reverb_server
$  python3 -m circuit_training.learning.ppo_reverb_server \
   --root_dir=${ROOT_DIR}  --port=${REVERB_PORT}

# Starts the Training job
# Change to the tmux session `train_job`.
# `ctrl + b` followed by `s`
$  python3 -m circuit_training.learning.train_ppo \
  --root_dir=${ROOT_DIR} \
  --replay_buffer_server_address=${REVERB_SERVER} \
  --variable_container_server_address=${REVERB_SERVER} \
  --num_episodes_per_iteration=16 \
  --global_batch_size=64 \
  --netlist_file=${NETLIST_FILE} \
  --init_placement=${INIT_PLACEMENT}

# Starts the Collect job
# Change to the tmux session `collect_job_00`.
# `ctrl + b` followed by `s`
$  python3 -m circuit_training.learning.ppo_collect \
  --root_dir=${ROOT_DIR} \
  --replay_buffer_server_address=${REVERB_SERVER} \
  --variable_container_server_address=${REVERB_SERVER} \
  --task_id=0 \
  --netlist_file=${NETLIST_FILE} \
  --init_placement=${INIT_PLACEMENT}

# Starts the Eval job
# Change to the tmux session `eval_job`.
# `ctrl + b` followed by `s`
$  python3 -m circuit_training.learning.eval \
  --root_dir=${ROOT_DIR} \
  --variable_container_server_address=${REVERB_SERVER} \
  --netlist_file=${NETLIST_FILE} \
  --init_placement=${INIT_PLACEMENT}

# Start Tensorboard.
# Change to the tmux session `tb_job`.
# `ctrl + b` followed by `s`
$  tensorboard dev upload --logdir ./logs

# <Optional>: Starts 2 more collect jobs to speed up training.
# Change to the tmux session `collect_job_01`.
# `ctrl + b` followed by `s`
$  python3 -m circuit_training.learning.ppo_collect \
  --root_dir=${ROOT_DIR} \
  --replay_buffer_server_address=${REVERB_SERVER} \
  --variable_container_server_address=${REVERB_SERVER} \
  --task_id=1 \
  --netlist_file=${NETLIST_FILE} \
  --init_placement=${INIT_PLACEMENT}

# Change to the tmux session `collect_job_02`.
# `ctrl + b` followed by `s`
$  python3 -m circuit_training.learning.ppo_collect \
  --root_dir=${ROOT_DIR} \
  --replay_buffer_server_address=${REVERB_SERVER} \
  --variable_container_server_address=${REVERB_SERVER} \
  --task_id=2 \
  --netlist_file=${NETLIST_FILE} \
  --init_placement=${INIT_PLACEMENT}

```

<a id='Results'></a>
## Results

The results below are for from scratch training, since the pre-trained policy
cannot be shared at this time.

### Ariane RISC-V CPU

View the full details of the Ariane experiment on our [details page](./docs/ARIANE.md).
With this code we are able to get similar results using from scratch training as
the fine-tuned results published in the
[paper](https://www.nature.com/articles/s41586-021-03544-w.epdf?sharing_token=tYaxh2mR5EozfsSL0WHZLdRgN0jAjWel9jnR3ZoTv0PW0K0NmVrRsFPaMa9Y5We9O4Hqf_liatg-lvhiVcYpHL_YQpqkurA31sxqtmA-E1yNUWVMMVSBxWSp7ZFFIWawYQYnEXoBE4esRDSWqubhDFWUPyI5wK_5B_YIO-D_kS8%3D).
At the time the paper was published, the fine-tuned results were better than the
from scratch result for a block of Ariane RISC-V. Improvements to the code have
resulted in 50% less resources used and a 2x walltime speedup in from scratch
training. The results below using this code base are the mean of 9 runs with 3
different seeds. This deviates from the papers 8 runs with 8 different seeds.
This approach was used to transparently show variation between seeds and run to
run with the same seed.

|| Proxy Wirelength | Proxy Congestion | Proxy Density |
|------|------------|------------------|---------------|
| **mean** | 0.1013     | 0.9174           | 0.5502
| **std**  | 0.0036     | 0.0647           | 0.0568


Summary of the [Paper](https://www.nature.com/articles/s41586-021-03544-w.epdf?sharing_token=tYaxh2mR5EozfsSL0WHZLdRgN0jAjWel9jnR3ZoTv0PW0K0NmVrRsFPaMa9Y5We9O4Hqf_liatg-lvhiVcYpHL_YQpqkurA31sxqtmA-E1yNUWVMMVSBxWSp7ZFFIWawYQYnEXoBE4esRDSWqubhDFWUPyI5wK_5B_YIO-D_kS8%3D)
results from 8 runs with 8 different using fine-tuning.

|| Proxy Wirelength | Proxy Congestion | Proxy Density |
|------|------------|------------------|---------------|
| **mean** | 0.1198     | 0.9718           | 0.5729
| **std**  | 0.0019     | 0.0346           | 0.0086


<a id='Testing'></a>
## Testing

```shell
# Runs tests with nightly TF-Agents.
$  tox -e py37,py38,py39
# Runs with latest stable TF-Agents.
$  tox -e py37-nightly,py38-nightly,py39-nightly

# Using our Docker for CI.
## Build the docker
$  docker build --tag circuit_training:ci -f tools/docker/ubuntu_ci tools/docker/
## Runs tests with nightly TF-Agents.
$  docker run -it --rm -v $(pwd):/workspace --workdir /workspace circuit_training:ci \
     tox -e py37-nightly,py38-nightly,py39-nightly
## Runs tests with latest stable TF-Agents.
$  docker run -it --rm -v $(pwd):/workspace --workdir /workspace circuit_training:ci \
     tox -e py37,py38,py39

```

<a id='Releases'></a>
## Releases

While we recommend running at `HEAD` we have tagged the code base to mark
compatibility with stable releases of the underlying libraries.

Release | Branch / Tag                                               | TF-Agents
------- | ---------------------------------------------------------- | ------------------
HEAD    | [main](https://github.com/google-research/circuit-training) | tf-agents-nightly
0.0.1   | [v0.0.1](https://github.com/google-research/circuit-training/tree/v0.0.1) | tf-agents==0.11.0

Follow this pattern to utilize the tagged releases:

```shell
$  git clone https://github.com/google-research/circuit-training.git
$  cd circuit-training
# Checks out the tagged version listed in the table in the releases section.
$  git checkout v0.0.1
# Installs the corresponding version of tf-agents along with Reverb and
# Tensorflow from the table.
$  pip install tf-agents[reverb]==x.x.x
# Copies the placement cost binary to /usr/local/bin and makes it executable.
$  sudo curl https://storage.googleapis.com/rl-infra-public/circuit-training/placement_cost/plc_wrapper_main \
     -o  /usr/local/bin/plc_wrapper_main
$  sudo chmod 555 /usr/local/bin/plc_wrapper_main
```

<a id='Contributing'></a>
## How to contribute

We're eager to collaborate with you! See [CONTRIBUTING](CONTRIBUTING.md)
for a guide on how to contribute. This project adheres to TensorFlow's
[code of conduct](CODE_OF_CONDUCT.md). By participating, you are expected to
uphold this code of conduct.


<a id='Principles'></a>

## Principles
This project adheres to [Google's AI principles](PRINCIPLES.md). By
participating, using or contributing to this project you are expected to adhere
to these principles.

<a id='Contributors'></a>

## Main Contributors
We would like to recognize the following individuals for their code
contributions, discussions, and other work to make the release of the Circuit
Training library possible.

* Sergio Guadarrama
* Summer Yue
* Ebrahim Songhori
* Joe Jiang
* Toby Boyd
* Azalia Mirhoseini
* Anna Goldie
* Mustafa Yazgan
* Shen Wang
* Terence Tam
* Young-Joon Lee
* Roger Carpenter
* Quoc Le
* Ed Chi


<a id='Citation'></a>

## How to cite

If you use this code, please cite both:

```
@article{mirhoseini2021graph,
  title={A graph placement methodology for fast chip design},
  author={Mirhoseini, Azalia and Goldie, Anna and Yazgan, Mustafa and Jiang, Joe Wenjie and Songhori, Ebrahim and Wang, Shen and Lee, Young-Joon and Johnson, Eric and Pathak, Omkar and Nazi, Azade and others},
  journal={Nature},
  volume={594},
  number={7862},
  pages={207--212},
  year={2021},
  publisher={Nature Publishing Group}
}
```


```
@misc{CircuitTraining2021,
  title = {{Circuit Training}: An open-source framework for generating chip
  floor plans with distributed deep reinforcement learning.},
  author = {Guadarrama, Sergio and Yue, Summer and Boyd, Toby and Jiang, Joe
  and Songhori, Ebrahim and Tam, Terence and Mirhoseini, Azalia},
  howpublished = {\url{https://github.com/google_research/circuit_training}},
  url = "https://github.com/google_research/circuit_training",
  year = 2021,
  note = "[Online; accessed 21-December-2021]"
}
```

<a id='Disclaimer'></a>

## Disclaimer

This is not an official Google product.

