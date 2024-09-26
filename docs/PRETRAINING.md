# Pre-Training Instruction

Pre-training a model requires a set of netlists for both training and for
evaluation. The model will be trained on all of the pre-training netlists
simultaneously and evaluated on the evaluation netlist(s). From an optimization
perspective, pre-training and fine-tuning are equivalent, and the same code is
used to perform both. The difference is that pre-training involves multiple
netlists and is performed prior to fine-tuning, whereas fine-tuning optimizes a
single target netlist. (For an example of fine-tuning on a single block, please
see [How to run training for Ariane RISC-V](./ARIANE.md).)

We provide an example below to show how to perform pre-training on multiple
netlists.

> Note: You must assign one or more collect jobs for each netlist in the
> pre-training set and one eval job for each evaluation netlist.

To start, let's assume we have the following list of netlists:

Train:

-   (netlist_0, init_placement_0)
-   (netlist_1, init_placement_1)
-   (netlist_2, init_placement_2)

Eval:

-   (netlist_3, init_placement_3)

The script below shows how to pass the netlists to the jobs. Please note that
the order of the netlists that are passed to the train job should be same as the
`netlist_index` provided to each collect job. The rest of arguments for the jobs
should follow the example given for [Ariane RISC-V](./ARIANE.md).
`max_sequence_length` should be the maximum `sequence_length` (number of macros)
for all the netlists.

## Train Job

```shell
$ python -m circuit_training.learning.train_ppo \
  --netlist_file=netlist_0 \
  --init_placement=init_placement_0 \
  --netlist_file=netlist_1 \
  --init_placement=init_placement_1 \
  --netlist_file=netlist_2 \
  --init_placement=init_placement_2 \
  --sequence_length=max_sequence_length
  --gin_bindings="CircuittrainingPPOLearner.allow_variable_length_episodes=True"
```

## Reverb Replay buffer Job:

```shell
$ python -m circuit_training.learning.ppo_reverb_server \
  --num_netlists=3
```

## Collect Jobs:

You need to run 3 sets of collect jobs for each netlist:

### Netlist_0 jobs:

```shell
$ python -m circuit_training.learning.ppo_collect \
  --netlist_file=netlist_0 \
  --init_placement=init_placement_0 \
  --netlist_index=0
```

### Netlist_1 jobs:

```shell
$ python -m circuit_training.learning.ppo_collect \
  --netlist_file=netlist_1 \
  --init_placement=init_placement_1 \
  --netlist_index=1
```

### Netlist_2 jobs:

```shell
$ python -m circuit_training.learning.ppo_collect \
  --netlist_file=netlist_2 \
  --init_placement=init_placement_2 \
  --netlist_index=2
```

## Eval Jobs:

```shell
$ python -m circuit_training.learning.eval \
  --netlist_file=netlist_3 \
  --init_placement=init_placement_3 \
```
