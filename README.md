<!--* freshness: {
  owner: 'agoldie'
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

*[A graph placement methodology for fast chip design.](https://www.nature.com/articles/s41586-021-03544-w)
Azalia Mirhoseini, Anna Goldie, Mustafa Yazgan, Joe Wenjie Jiang, Ebrahim
Songhori, Shen Wang, Young-Joon Lee, Eric Johnson, Omkar Pathak, Azade Nazi,
Jiwoo Pak, Andy Tong, Kavya Srinivasa, William Hang, Emre Tuncer, Quoc V. Le,
James Laudon, Richard Ho, Roger Carpenter & Jeff Dean, 2021. Nature, 594(7862),
pp.207-212.
[[PDF]](https://www.nature.com/articles/s41586-021-03544-w.epdf?sharing_token=tYaxh2mR5EozfsSL0WHZLdRgN0jAjWel9jnR3ZoTv0PW0K0NmVrRsFPaMa9Y5We9O4Hqf_liatg-lvhiVcYpHL_YQpqkurA31sxqtmA-E1yNUWVMMVSBxWSp7ZFFIWawYQYnEXoBE4esRDSWqubhDFWUPyI5wK_5B_YIO-D_kS8%3D)*

Our hope is that *Circuit Training* will foster further collaborations between
academia and industry, and enable advances in deep reinforcement learning for
Electronic Design Automation, as well as, general combinatorial and decision
making optimization problems. Capable of optimizing chip blocks with hundreds of
macros, *Circuit Training* automatically generates floor plans in hours, whereas
baseline methods often require human experts in the loop and can take months.

Circuit training is built on top of
[TF-Agents](https://github.com/tensorflow/agents) and
[TensorFlow 2.x](https://www.tensorflow.org/) with support for eager execution,
distributed training across multiple GPUs, and distributed data collection
scaling to 100s of actors.

## Table of contents

<a href='#Features'>Features</a><br>
<a href='#Installation'>Installation</a><br>
<a href='#QuickStart'>Quick start</a><br>
<a href='#Results'>Results</a><br>
<a href='#Testing'>Testing</a><br>
<a href='#Releases'>Releases</a><br>
<a href='#FAQ'>FAQ</a><br>
<a href='#Contributing'>How to contribute</a><br>
<a href='#Principles'>AI Principles</a><br>
<a href='#Contributors'>Contributors</a><br>
<a href='#Citation'>How to cite</a><br>
<a href='#Disclaimer'>Disclaimer</a><br>

<a id='Features'></a>

## Features

*   Places netlists with hundreds of macros and millions of stdcells (in
    clustered format).
*   Computes both macro location and orientation (flipping).
*   Optimizes multiple objectives including wirelength, congestion, and density.
*   Supports alignment of blocks to the grid, to model clock strap or macro
    blockage.
*   Supports macro-to-macro, macro-to-boundary spacing constraints.
*   Allows users to specify their own technology parameters, e.g. and routing
    resources (in routes per micron) and macro routing allocation.
*   Generates [clustered netlists](https://github.com/google-research/circuit_training/tree/main/circuit_training/grouping).
*   **[Update 11-JULY-2022]** Working with vendors for approval to release
    tcl scripts for major EDA tools (Innovus, ICC2) that generate the
    [Netlist Protocol Buffer](https://github.com/google-research/circuit_training/blob/main/docs/NETLIST_FORMAT.md)
    used as the input for circuit training.
    [Issue #3](https://github.com/google-research/circuit_training/issues/3)

<a id='Installation'></a>

## Installation

Circuit Training requires:

*   Installing TF-Agents which includes Reverb and TensorFlow.
*   Downloading the placement cost binary into your system path.
*   Downloading the circuit-training code.

Using the code at `HEAD` with the nightly release of TF-Agents is recommended.

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
[tensorboard](https://tensorboard.dev/experiment/r1Xn1pD3SGKTGyo64saeaw). The
full scale Ariane RISC-V experiment matching the paper is detailed in
[Circuit training for Ariane RISC-V](./docs/ARIANE.md).

The following jobs will be created by the steps below:

*   1 Replay Buffer (Reverb) job
*   1-3 Collect jobs
*   1 Train job
*   1 Eval job

Each job is started in a `tmux` session. To switch between sessions use `ctrl +
b` followed by `s` and then select the specified session.

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

The results below are reported for training from scratch, since the pre-trained
model cannot be shared at this time.

### Ariane RISC-V CPU

View the full details of the Ariane experiment on our
[details page](./docs/ARIANE.md). With this code we are able to get comparable
or better results training from scratch as fine-tuning a pre-trained model. At
the time the paper was published, training from a pre-trained model resulted in
better results than training from scratch for the Ariane RISC-V. Improvements to
the code have also resulted in 50% less GPU resources needed and a 2x walltime
speedup even in training from scratch. Below are the mean and standard deviation
for 3 different seeds run 3 times each. This is slightly different than what was
used in the paper (8 runs each with a different seed), but better captures the
different sources of variability.

         | Proxy Wirelength | Proxy Congestion | Proxy Density
-------- | ---------------- | ---------------- | -------------
**mean** | 0.1013           | 0.9174           | 0.5502
**std**  | 0.0036           | 0.0647           | 0.0568

The table below summarizes the
[paper](https://www.nature.com/articles/s41586-021-03544-w.epdf?sharing_token=tYaxh2mR5EozfsSL0WHZLdRgN0jAjWel9jnR3ZoTv0PW0K0NmVrRsFPaMa9Y5We9O4Hqf_liatg-lvhiVcYpHL_YQpqkurA31sxqtmA-E1yNUWVMMVSBxWSp7ZFFIWawYQYnEXoBE4esRDSWqubhDFWUPyI5wK_5B_YIO-D_kS8%3D)
result for fine-tuning from a pre-trained model over 8 runs with each one using
a different seed.

         | Proxy Wirelength | Proxy Congestion | Proxy Density
-------- | ---------------- | ---------------- | -------------
**mean** | 0.1198           | 0.9718           | 0.5729
**std**  | 0.0019           | 0.0346           | 0.0086

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

While we recommend running at `HEAD`, we have tagged the code base to mark
compatibility with stable releases of the underlying libraries.

Release | Branch / Tag                                                              | TF-Agents
------- | ------------------------------------------------------------------------- | ---------
HEAD    | [main](https://github.com/google-research/circuit-training)               | tf-agents-nightly
0.0.1   | [v0.0.1](https://github.com/google-research/circuit-training/tree/v0.0.1) | tf-agents==0.11.0

Follow this pattern to utilize the tagged releases:

```shell
$  git clone https://github.com/google-research/circuit-training.git
$  cd circuit-training
# Checks out the tagged version listed in the table in the releases section.
$  git checkout v0.0.1
# Installs the corresponding version of TF-Agents along with Reverb and
# Tensorflow from the table.
$  pip install tf-agents[reverb]==x.x.x
# Copies the placement cost binary to /usr/local/bin and makes it executable.
$  sudo curl https://storage.googleapis.com/rl-infra-public/circuit-training/placement_cost/plc_wrapper_main \
     -o  /usr/local/bin/plc_wrapper_main
$  sudo chmod 555 /usr/local/bin/plc_wrapper_main
```

<a id='FAQ'></a>

## Frequently Asked Questions

We wrote this FAQ to answer frequently asked questions about our work. Please
reach out to us if you have any other questions!

#### What is the goal and philosophy of our team?

Our goal is to help chip designers do their jobs better and faster, and we
welcome any method that moves us in that direction. To ensure that we are
solving real world problems, we work closely with chip designers to understand
and address their needs.

#### What is the impact of our work?

To our knowledge, this is the first deep reinforcement learning (RL) method used
in production to design hardware products. More specifically, the RL method
described in the Nature paper generated macro placements that were frozen and
taped out in Google’s AI accelerator chip (TPU-v5).

We are also excited to see that top EDA and chip design companies (e.g.
[Synopsys](https://www.forbes.com/sites/moorinsights/2020/04/20/using-ai-to-build-better-chips/?sh=63551aef306c),
[Cadence](https://www.zdnet.com/article/ai-on-the-bench-cadence-offers-machine-learning-to-smooth-chip-design/),
[NVIDIA](https://research.nvidia.com/publication/2021-07_NVCell%3A-Standard-Cell),
etc.) have announced initiatives to use similar RL-based methods in their tools
and chip design efforts.

#### Have we evaluated our method on open-source benchmarks?

We are focused on modern sub-10nm chips like TPU and Pixel, but we did publish
an article in MLCAD 2021 led by Prof. David Pan and his student Zixuan Jiang,
where we report results on the open-source ISPD 2015 benchmarks after unfixing
macros. In any case, we have open-sourced our method, so the community is free
to try it out on any benchmark.

#### How do we compare to commercial autoplacers?

Due to licensing agreements, we cannot publish any public comparison with
commercial autoplacers. However, we can say that our strongest baseline is the
physical design team working directly with the assistance of commercial
autoplacers, and we outperform this baseline (see “manual” baseline in Table 1
of our Nature article).

#### How do we perform clustering of standard cells?

In our Nature paper, we describe how to use hMETIS to cluster standard cells,
including all necessary settings. For detailed settings, please see Extended
Data Table 3 from our [Nature article](http://rdcu.be/cmedX). Internally, Google
pays for a commercial license, but non-commercial entities are welcome to use a
free open-source license

Regardless, our method runs on unclustered netlists as well, so you can skip the
preprocessing step if you wish, though we’ve found clustering to benefit both
our RL method and baseline placers. The complexity of our method scales with the
number of macros, not the number of standard cells, so the runtime will not be
overly affected.

#### What netlist formats do we support?

Our placer represents netlists in the open-source
[protocol buffer](https://developers.google.com/protocol-buffers) format. You
can learn more about the format [here](./docs/NETLIST_FORMAT.md). To run on
netlists in other formats (e.g. LEF/DEF or Bookshelf), you can convert to
protocol buffer format. Please see our
[quick start guide](https://github.com/google-research/circuit_training#QuickStart)
for an example of how to use this format on the open-source RISC-V Ariane CPU.

#### Why do we claim “fast chip design” when RL is slower than analytic solvers?

When we say “fast”, we mean that we actually help chip designers do their jobs
faster, not that our algorithm runs fast per se. Our method can, in hours, do
what a human chip designer needs weeks or months to perform.

If an analytic method optimizes for wirelength and produces a result in ~1
minute, that’s obviously faster than hours of RL optimization; however, if the
result does not meet design criteria and therefore physical design experts must
spend weeks further iterating in the loop with commercial EDA tools, then it’s
not faster in any way that matters.

#### In our Nature experiments, why do we report QoR metrics rather than wirelength alone?

Our goal is to develop methods that help chip designers do their job better and
faster. We therefore designed the experiments in our paper to mimic the true
production setting as closely as possible, and report QoR (Quality of Result)
metrics.

QoR metrics can take up to 72 hours to generate with a commercial EDA tool, but
are highly accurate measurements of all key metrics, including wirelength,
horizontal/vertical congestion, timing (TNS and WNS), power, and area.

QoR metrics are closest to physical ground truth and are used by production chip
design teams to decide which placements are sent for manufacturing. In contrast,
proxy costs like approximate wirelength and congestion can be computed cheaply
and are useful for optimization, but are not used to make real world decisions
as they can vary significantly from QoR.

It is also worth noting that metrics like wirelength and routing congestion
directly trade off against each other (e.g. placing nodes close to one another
increases congestion, but reduces wirelength), so optimizing or evaluating for
wirelength alone is unlikely to result in manufacturable chip layouts.

#### In our Nature experiments, do we perform any postprocessing on the RL results?

No. In our Nature experiments, we do not apply any postprocessing to the RL
results.

In our open-source code, we provide an optional 1-5 minute coordinate descent
postprocessing step, which we found to slightly improve wirelength. You are
welcome to turn it on or off with a flag, and to compare performance with or
without it.

#### What was the process for open-sourcing this code?

Open-sourcing our code involved partnering with another team at Google
([TF-Agents](https://www.tensorflow.org/agents)). TF-Agents first replicated the
results in our Nature article using our codebase, then reimplemented our method
and replicated our results using their own implementation, and then open-sourced
their implementation as it does not rely on any internal infrastructure.

Getting approval to open-source this code, ensuring compliance with export
control restrictions, migrating to TensorFlow 2.x, and removing dependencies
from all Google infrastructure was quite time-consuming; but we felt that it was
worth the effort to be able to share our method with the community.

<a id='Contributing'></a>

## How to contribute

We're eager to collaborate with you! See [CONTRIBUTING](CONTRIBUTING.md) for a
guide on how to contribute. This project adheres to TensorFlow's
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

*   Sergio Guadarrama
*   Summer Yue
*   Ebrahim Songhori
*   Joe Jiang
*   Toby Boyd
*   Azalia Mirhoseini
*   Anna Goldie
*   Mustafa Yazgan
*   Shen Wang
*   Terence Tam
*   Young-Joon Lee
*   Roger Carpenter
*   Quoc Le
*   Ed Chi

<a id='Citation'></a>

## How to cite

If you use this code, please cite both:

```
@article{mirhoseini2021graph,
  title={A graph placement methodology for fast chip design},
  author={Mirhoseini, Azalia and Goldie, Anna and Yazgan, Mustafa and Jiang, Joe
  Wenjie and Songhori, Ebrahim and Wang, Shen and Lee, Young-Joon and Johnson,
  Eric and Pathak, Omkar and Nazi, Azade and Pak, Jiwoo and Tong, Andy and
  Srinivasa, Kavya and Hang, William and Tuncer, Emre and V. Le, Quoc and
  Laudon, James and Ho, Richard and Carpenter, Roger and Dean, Jeff},
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
  Wenjie and Songhori, Ebrahim and Tam, Terence and Mirhoseini, Azalia},
  howpublished = {\url{https://github.com/google_research/circuit_training}},
  url = "https://github.com/google_research/circuit_training",
  year = 2021,
  note = "[Online; accessed 21-December-2021]"
}
```

<a id='Disclaimer'></a>

## Disclaimer

This is not an official Google product.
