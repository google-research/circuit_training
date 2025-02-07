<!--* freshness: {
  owner: 'wenjiej'
  owner: 'esonghori'
  owner: 'sguada'
  owner: 'morpheus-core'
  reviewed: '2024-08-14'
  review_interval: '12 months'
} *-->

# AlphaChip: An open-source framework for generating chip floorplans with distributed deep reinforcement learning.

*AlphaChip* is an open-source framework
for generating chip floorplans with distributed deep reinforcement learning.
This framework reproduces the methodology published in the Nature 2021 paper:

*[A graph placement methodology for fast chip design.](https://www.nature.com/articles/s41586-021-03544-w)
Azalia Mirhoseini, Anna Goldie, Mustafa Yazgan, Joe Wenjie Jiang, Ebrahim
Songhori, Shen Wang, Young-Joon Lee, Eric Johnson, Omkar Pathak, Azade Nazi,
Jiwoo Pak, Andy Tong, Kavya Srinivasa, William Hang, Emre Tuncer, Quoc V. Le,
James Laudon, Richard Ho, Roger Carpenter & Jeff Dean, 2021. Nature, 594(7862),
pp.207-212.
[[PDF]](https://www.nature.com/articles/s41586-021-03544-w.epdf?sharing_token=tYaxh2mR5EozfsSL0WHZLdRgN0jAjWel9jnR3ZoTv0PW0K0NmVrRsFPaMa9Y5We9O4Hqf_liatg-lvhiVcYpHL_YQpqkurA31sxqtmA-E1yNUWVMMVSBxWSp7ZFFIWawYQYnEXoBE4esRDSWqubhDFWUPyI5wK_5B_YIO-D_kS8%3D)*

AlphaChip--one of the first reinforcement learning approaches used to solve a
real-world engineering problem--has led to a proliferation of research in AI
for chips over the past few years. It is now used to design layouts for chips
across Alphabet and outside, and has been extended to various stages of the
design process, including logic synthesis, macro selection, timing optimization,
and more! We hope that researchers will continue building on top of AlphaChip
methodologies and open-source framework. Please see our [blogpost](https://deepmind.google/discover/blog/how-alphachip-transformed-computer-chip-design/) for more information.

AlphaChip is built on top of
[TF-Agents](https://github.com/tensorflow/agents) and
[TensorFlow 2.x](https://www.tensorflow.org/) with support for eager execution,
distributed training across multiple GPUs, and distributed data collection
scaling to 100s of actors.

## Table of contents

<a href='#Features'>Features</a><br>
<a href='#Installation'>Installation</a><br>
<a href='#QuickStart'>Quick start</a><br>
<a href='#Testing'>Testing</a><br>
<a href='#Releases'>Releases</a><br>
<a href='#PreTrainedModelCheckpoint'>Pre-Trained Model Checkpoint</a><br>
<a href='#HowToUseTheCheckpoint'>How to use the checkpoint</a><br>
<a href='#Results'>Results</a><br>
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
*   Supports fixed macros.
*   Supports [DREAMPlace](https://github.com/limbo018/DREAMPlace) as the stdcell placer.
*   Allows users to specify their own technology parameters, e.g. and routing
    resources (in routes per micron) and macro routing allocation.
*   Generates
    [clustered netlists](https://github.com/google-research/circuit_training/tree/main/circuit_training/grouping).
*   [TILOS-AI-Institute](https://www.tilos.ai/) has created a
    [script](https://github.com/TILOS-AI-Institute/MacroPlacement/tree/main/CodeElements/FormatTranslators)
    to convert LEF/DEF and Bookshelf to the
    [Netlist Protocol Buffer](https://github.com/google-research/circuit_training/blob/main/docs/NETLIST_FORMAT.md)
    used as the input for AlphaChip.

<a id='Installation'></a>

## Installation

> :warning: AlphaChip only supports Linux based OSes.

> :warning: AlphaChip requires Python 3.9 or greater.

## Stable

AlphaChip is a research project. We are not currently creating PyPi
builds. Stable in this instance is relative to HEAD and means that the code
was tested at this point in time and branched. With upstream libraires
constantly changing; older branches may end up rotting faster than expected.

The steps below install the most recent branch and the archive is in the
[releases section](#releases). There are two methods for installing; but before
doing either one you need to run the preliminary setup](#preliminary-setup).

*  [Use the docker](#using-the-docker) (**Highly Recommended**)
*  [Install locally](#install-locally)


### Preliminary Setup

Before following the instructions set the following variables and clone the
repo:

```shell
$ export CT_VERSION=0.0.4
# Currently supports python3.9, python3.10, and python3.11
# The docker is python3.9 only.
$ export PYTHON_VERSION=python3.9
$ export DREAMPLACE_PATTERN=dreamplace_20231214_c5a83e5_${PYTHON_VERSION}.tar.gz
# If the verson of TF-Agents in the table is not current, change this command to
# match the version tf-agenst that matches the branch of AlphaChip used.
$ export TF_AGENTS_PIP_VERSION=tf-agents[reverb]

# Clone the Repo and checkout the desired branch.
$  git clone https://github.com/google-research/circuit_training.git
$  git -C $(pwd)/circuit_training checkout r${CT_VERSION}
```

### Using the docker

Do not forget to do the [prelimary setup](#preliminary-setup). The cleanest way
to use AlphaChip is to use the docker, these commands will create a
docker with all the dependencies needed:

```shell
$ export REPO_ROOT=$(pwd)/circuit_training

# Build the docker image.
$ docker build --pull --no-cache --tag circuit_training:core \
    --build-arg tf_agents_version="${TF_AGENTS_PIP_VERSION}" \
    --build-arg dreamplace_version="${DREAMPLACE_PATTERN}" \
    --build-arg placement_cost_binary="plc_wrapper_main_${CT_VERSION}" \
    -f "${REPO_ROOT}"/tools/docker/ubuntu_circuit_training ${REPO_ROOT}/tools/docker/

# Run the end2end smoke test using the image. Takes 10-20 minutes.
$ mkdir -p ${REPO_ROOT}/logs
$ docker run --rm -v ${REPO_ROOT}:/workspace --workdir /workspace circuit_training:core \
    bash tools/e2e_smoke_test.sh --root_dir /workspace/logs
```

### Install locally

Do not forget to do the [prelimary setup](#preliminary-setup).

AlphaChip installation steps:

*   Install our DREAMPlace binary.
*   Install TF-Agents and The Placement Cost Binary
*   Run a test


#### Install DREAMPlace

Follow the [instructions](#install-dreamplace) for DREAMPlace but do not change
the ENV VARS that you already exported previously.


#### Install TF-Agents and the Placement Cost binary

These commands install TF-Agents and the placement cost binary.

```shell
# Installs TF-Agents with stable versions of Reverb and TensorFlow 2.x.
$  pip install $TF_AGENTS_PIP_VERSION
$  pip install tf-keras
# Using keras-2
$ export TF_USE_LEGACY_KERAS=1
# Copies the placement cost binary to /usr/local/bin and makes it executable.
$  sudo curl https://storage.googleapis.com/rl-infra-public/circuit-training/placement_cost/plc_wrapper_main_${CT_VERSION} \
     -o  /usr/local/bin/plc_wrapper_main
$  sudo chmod 555 /usr/local/bin/plc_wrapper_main
```

#### Run a test.

These commands run a basic unit test; if the current stable tf-agents is not the
version you installed, then edit the tox.ini file and change `tf-agents[reverb]`
to `tf-agents[reverb]~=<version you want>`

```shell
tox -e py39-stable -- circuit_training/grouping/grouping_test.py
```


## HEAD

We recommand using [stable](#stable) branches; but our team does work from the
`HEAD`. The main issue is `HEAD` breaks when upstream libraries are broken and
our `HEAD` utilizes other nightly created libraries adding to the variablity.

The steps below install the most recent branch and the archive is in the
[releases section](#releases). There are two methods for installing; but before
doing either one you need to run the [preliminary setup](#preliminary-setup-2).

*  [Use the docker](#using-the-docker-2) (**Highly Recommended**)
*  [Install locally](#install-locally-2)


### Preliminary Setup

Before following the instructions set the following variables and clone the
repo:

```shell
# Currently supports python3.9, python3.10, and python3.11
# The docker is python3.9 only.
$ export PYTHON_VERSION=python3.9
$ export DREAMPLACE_PATTERN=dreamplace_${PYTHON_VERSION}.tar.gz

# Clone the Repo and checkout the desired branch.
$  git clone https://github.com/google-research/circuit_training.git
```

### Using the docker

Do not forget to do the [preliminary setup](#preliminary-setup). The cleanest way
to use AlphaChip is to use docker, these commands will create an image
with all the dependencies needed:

```shell
$ export REPO_ROOT=$(pwd)/circuit_training

# Builds the image with current DREAMPlace and Placement Cost Binary.
$ docker build --pull --no-cache --tag circuit_training:core \
    --build-arg tf_agents_version="tf-agents-nightly[reverb]" \
    -f "${REPO_ROOT}"/tools/docker/ubuntu_circuit_training ${REPO_ROOT}/tools/docker/

# Run the end2end smoke test using the image. Takes 10-20 minutes.
$ mkdir -p ${REPO_ROOT}/logs
$ docker run --rm -v ${REPO_ROOT}:/workspace --workdir /workspace circuit_training:core \
    bash tools/e2e_smoke_test.sh --root_dir /workspace/logs
```

### Install locally

AlphaChip installation steps:

*   Install our DREAMPlace binary.
*   Install TF-Agents Nightly and the placement cost binary
*   Run a test

#### Install DREAMPlace

Follow the [instructions](#install-dreamplace) for DREAMPlace but do not change
the ENV VARS that you already exported previously.


#### Install TF-Agents and the Placement Cost binary

These commands install TF-Agents and the placement cost binary.

```shell
# Installs TF-Agents with stable versions of Reverb and TensorFlow 2.x.
$  pip install tf-agents-nightly[reverb]
$  pip install tf-keras
# Using keras-2
$ export TF_USE_LEGACY_KERAS=1
# Copies the placement cost binary to /usr/local/bin and makes it executable.
$  sudo curl https://storage.googleapis.com/rl-infra-public/circuit-training/placement_cost/plc_wrapper_main \
     -o  /usr/local/bin/plc_wrapper_main
$  sudo chmod 555 /usr/local/bin/plc_wrapper_main
```

#### Run a test.

These commands run a basic unit test.

```shell
tox -e py39-nightly -- circuit_training/grouping/grouping_test.py
```

## Install DREAMPlace

DREAMPlace is **not** provided as a PyPi package and needs to be compiled. We 
provide compiled versions of DREAMPlace taken from our
[branch](https://github.com/esonghori/DREAMPlace/tree/circuit_training) for a
range of Python versions built for our docker image (Ubuntu 20.4). We also use
them for presubmit testing. If our binaries are not compatible with your OS tool
chain, you will need to compile your own version. We use this
[script](https://github.com/google-research/circuit_training/blob/main/tools/bootstrap_dreamplace_build.sh)
to create our DREAMPlace binary.

```shell
# These ENV VARS may have been set above, do not export again if already set.
$ export PYTHON_VERSION=python3.9
$ export DREAMPLACE_PATTERN=dreamplace_${PYTHON_VERSION}.tar.gz
# Installs DREAMPlace into `/dreamplace`. Anywhere is fine as long as PYTHONPATH
# is set correctly.
$  mkdir -p /dreamplace
# Picks the binary that matches your version of Python.
$  curl https://storage.googleapis.com/rl-infra-public/circuit-training/dreamplace/dreamplace_python3.9.tar.gz -o /dreamplace/dreamplace.tar.gz

# Unpacks the package.
$  tar xzf /dreamplace/dreamplace.tar.gz -C /dreamplace/

# Sets the python path so we can find Placer with `import dreamplace.Placer`
# This also needs to put all of DREAMPlace at the root because DREAMPlace python
# is not setup like a package with imports like `dreamplace.Param`.
$  export PYTHONPATH="${PYTHONPATH}:/dreamplace:/dreamplace/dreamplace"

# DREAMPlace requires some additional system and python libraries
# System packages
$  apt-get install -y \
      flex \
      libcairo2-dev \
      libboost-all-dev

# Python packages
$  python3 -mpip install pyunpack>=0.1.2 \
      patool>=1.12 \
      timeout-decorator>=0.5.0 \
      matplotlib>=2.2.2 \
      cairocffi>=0.9.0 \
      pkgconfig>=1.4.0 \
      setuptools>=39.1.0 \
      scipy>=1.1.0 \
      numpy>=1.15.4 \
      torch==1.13.1 \
      shapely>=1.7.0

```

<a id='QuickStart'></a>

## Quick start

The best quick start is to run the
[end2end smoke test](https://github.com/google-research/circuit_training/tree/main/tools#end-to-end-smoke-test)
and then look at the full distributed example
[AlphaChip for Ariane RISC-V](./docs/ARIANE.md).
For the pre-training
on multiple netlists see [Pre-Training Instruction](./docs/PRETRAINING.md).

<a id='Testing'></a>

## Testing

```shell
# Runs tests with nightly TF-Agents.
$  tox -e py39-nightly,py310-nightly,py311-nightly
# Runs with latest stable TF-Agents.
$  tox -e py39-stable,py310-stable,py311-stable

# Using our Docker for CI.
## Build the docker
$  docker build --tag circuit_training:ci -f tools/docker/ubuntu_ci tools/docker/
## Runs tests with nightly TF-Agents.
$  docker run -it --rm -v $(pwd):/workspace --workdir /workspace circuit_training:ci \
     tox -e py39-nightly,py310-nightly,py311-nightly
## Runs tests with latest stable TF-Agents.
$  docker run -it --rm -v $(pwd):/workspace --workdir /workspace circuit_training:ci \
     tox -e py39-stable,py310-stable,py311-stable

```

<a id='Releases'></a>

## Releases

While running at `HEAD` likely works, working from a branch has advantages of
being more stable. We have tagged the code base to mark
compatibility with stable releases of the underlying libraries. For DREAMPlace
the filename pattern can be used to install DREAMPle for the versions of Python
supported. For the Placement Cost binary, the ULR is to the version of the PLC
used at the time the branch was cut.

Release | Branch / Tag                                                              | TF-Agents                 | DREAMPlace                       | PL
------- | ------------------------------------------------------------------------- | ------------------------- | -------------------------------- | -------------- |
HEAD    | [main](https://github.com/google-research/circuit-training)               | tf-agents-nightly[reverb] |
0.0.4   | [v0.0.4](https://github.com/google-research/circuit_training/tree/r0.0.4) | tf-agents[reverb]~=0.19.0 | dreamplace_20231214_c5a83e5_python3.9.tar.gz | [plc_wrapper_main_0.0.4](https://storage.googleapis.com/rl-infra-public/circuit-training/placement_cost/plc_wrapper_main_0.0.4)
0.0.3   | [v0.0.3](https://github.com/google-research/circuit_training/tree/r0.0.3) | tf-agents[reverb]~=0.16.0 | dreamplace_20230414_b31e8af_python3.9.tar.gz | [plc_wrapper_main_0.0.3](https://storage.googleapis.com/rl-infra-public/circuit-training/placement_cost/plc_wrapper_main_0.0.3)
0.0.2   | [v0.0.2](https://github.com/google-research/circuit_training/tree/v0.0.2) | tf-agents[reverb]~=0.16.0 |

<a id='PreTrainedModelCheckpoint'></a>

## Pre-Trained Model Checkpoint

Unlike prior approaches, our method is a learning-based approach, meaning that
it becomes better and faster as it sees and solves more instances of the chip
placement problem. This pre-training step significantly improves its speed,
reliability, and placement quality, as discussed in the original Nature article
and a follow-up study at ISPD 2022 ([Summer Yue, Ebrahim Songhori, Joe Jiang,
Toby Boyd, Anna Goldie, Azalia Mirhoseini, Sergio Guadarrama. Scalability and
Generalization of Circuit Training for Chip Floorplanning. ISPD, 2022.`]
(https://dl.acm.org/doi/abs/10.1145/3505170.3511478)).

We release a model checkpoint pre-trained on 20 TPU blocks, which can
serve as a starting point for model training and fine-tuning purposes. Please
note that, like any other deep learning models (such as large language and
vision models), increasing the number of training examples and using
in-distribution data during pre-training will improve the quality of results.
Therefore, for best results, we strongly recommend
[pre-training on your own chip blocks](./docs/PRETRAINING.md), as these will
represent the most relevant placement experience for the RL agent.

Obviously, not performing any pre-training, i.e., training from scratch,
removes the RL agent's ability to learn from prior experience.


<a id='HowToUseTheCheckpoint'></a>

### How to use the checkpoint

First, download and untar the checkpoint:

```shell
sudo curl https://storage.googleapis.com/rl-infra-public/circuit-training/tpu_checkpoint_20240815.tar.gz -o $PWD/tpu_checkpoint_20240815.tar.gz
tar -xvf tpu_checkpoint_20240815.tar.gz
CHECKPOINT_DIR=$PWD/tpu_checkpoint_20240815/
```

Then, set the following flags in the train binary to the provided checkpoint
directory:

```shell
python3.9 -m circuit_training.learning.train_ppo \
  ... \
  --policy_checkpoint_dir=${CHECKPOINT_DIR} \
  --policy_saved_model_dir=${CHECKPOINT_DIR}
```

<a id='Results'></a>

## Results

The results below are reported for training from scratch, since the pre-trained
model cannot be shared at this time.

### Ariane RISC-V CPU

View the full details of the Ariane experiment on our
[details page](./docs/ARIANE.md). Improvements to the code have also resulted
in 50% less GPU resources needed and a 2x walltime speedup even in training
from scratch. Below are the mean and standard deviation for 3 different seeds
run 3 times each. This is slightly different than what was used in the paper
(8 runs each with a different seed), but better captures the different sources
of variability.

 Metric  | Proxy Wirelength | Proxy Congestion | Proxy Density
-------- | ---------------- | ---------------- | -------------
**mean** | 0.1013           | 0.9174           | 0.5502
**std**  | 0.0036           | 0.0647           | 0.0568

The table below summarizes the
[paper](https://www.nature.com/articles/s41586-021-03544-w.epdf?sharing_token=tYaxh2mR5EozfsSL0WHZLdRgN0jAjWel9jnR3ZoTv0PW0K0NmVrRsFPaMa9Y5We9O4Hqf_liatg-lvhiVcYpHL_YQpqkurA31sxqtmA-E1yNUWVMMVSBxWSp7ZFFIWawYQYnEXoBE4esRDSWqubhDFWUPyI5wK_5B_YIO-D_kS8%3D)
result for fine-tuning from a pre-trained model over 8 runs with each one using
a different seed.

 Metric  | Proxy Wirelength | Proxy Congestion | Proxy Density
-------- | ---------------- | ---------------- | -------------
**mean** | 0.1198           | 0.9718           | 0.5729
**std**  | 0.0019           | 0.0346           | 0.0086

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
contributions, discussions, and other work to make the release of the AlphaChip
library possible.

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
  author={Mirhoseini*, Azalia and Goldie*, Anna and Yazgan, Mustafa and Jiang, Joe
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
  Wenjie and Songhori, Ebrahim and Tam, Terence, Goldie, Anna and Mirhoseini,
  Azalia},
  howpublished = {\url{https://github.com/google_research/circuit_training}},
  url = "https://github.com/google_research/circuit_training",
  year = 2021,
  note = "[Online; accessed 21-December-2021]"
}
```

In addition, if you used the open-sourced checkpoint, please also cite:

```
@misc{AlphaChipCheckpoint2024,
  title = {A Pre-Trained Checkpoint for {AlphaChip}},
  author = {Jiang, Joe Wenjie and Songhori, Ebrahim and Mirhoseini, Azalia and
  Goldie, Anna and Guadarrama, Sergio and Yue, Summer and
  Boyd, Toby and Tam, Terence, and Wu, Guanhang and Lee, Kuang-Huei and
  Zhuang, Vincent and Yazgan, Mustafa and and Wang, Shen and Lee, Young-Joon and
  Johnson, Eric and Pathak, Omkar and Nazi, Azade and Pak, Jiwoo and
  Tong, Andy and Srinivasa, Kavya and Hang, William and Tuncer, Emre and
  V. Le, Quoc and Laudon, James and Ho, Richard and Carpenter, Roger and
  Dean, Jeff},
  howpublished = {\url{https://github.com/google-research/circuit_training/?tab=readme-ov-file#PreTrainedModelCheckpoint}},
  url = "https://github.com/google-research/circuit_training/?tab=readme-ov-file#PreTrainedModelCheckpoint",
  year = 2024,
  note = "[Online; accessed 25-September-2024]"
}
```

<a id='Disclaimer'></a>

## Disclaimer

This is not an official Google product.
