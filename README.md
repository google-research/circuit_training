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
making optimization problems. Capable of optimizing over hundreds of blocks,
*Circuit Training*  generates floorplans in hours instead of months. This would
have taken months to achieve manually.

Circuit training is built on top of [TF-Agents](https://github.com/tensorflow/agents)
and [TensorFlow 2.x](https://www.tensorflow.org/) with
support for eager execution, distributed training across multi-GPUs,
and distributed data collection scaling to 100s of actors running in
parallel sampled from a [Reverb](https://github.com/deepmind/reverb)
based replay buffer.

## Table of contents
<a href='#Features'>Main features</a><br>
<a href='#Installation'>How to install</a><br>
<a href='#Experiments'>How to run experiments</a><br>
<a href='#Hyperparameters'>Hyperparameters</a><br>
<a href='#Testing'>How to test</a><br>
<a href='#Releases'>Releases</a><br>
<a href='#Contributing'>How to contribute</a><br>
<a href='#Principles'>AI Principles</a><br>
<a href='#Contributors'>Contributors</a><br>
<a href='#Citation'>How to cite</a><br>
<a href='#Disclaimer'>Disclaimer</a><br>

<a id='Features'></a>

## Main features
* Places netlists with hundreds of macros and millions of stdcells (in clustered format).
* Computes both macro location and orientation (flipping).
* Optimizes multiple objectives including wirelength, congestion, and density.
* Supports blockages on the grid, to model clock strap or macro blockage.
* Supports macro-to-macro, macro-to-boundary spacing constraints.
* Allows users to specify their own technology parameters, e.g. and routing resources (in routes per micron) and macro routing allocation.
* **Coming soon**: Tools for generating a clustered netlist given a netlist in common formats (Bookshelf and LEF/DEF).
* **Coming soon**: Generates macro placement tcl command compatible with major EDA tools (Innovus, ICC2).


<a id='Installation'></a>
## How to install

<a id='Experiments'></a>
## How to run experiments


<a id='Testing'></a>
## How to run tests

<a id='Releases'></a>
## Releases


<a id='Hyperparameters'></a>
## Hyperparameters
Here, we have listed all the hyperparameters in the code and their default
value that we used for our experiments.
Some hyperparameters are changed from the paper to make the training more
stable for the Ariane block (The hyperparameters in the paper were set for
TPU blocks which have different characteristics).
For training, we use the clipping
version of proximal policy optimization (PPO)
(Schulman et al., 2017) without the KL divergence penalty implemented by
[tf-agents](https://www.tensorflow.org/agents).
The default for the training hyperparameters, if not specified in the table, is
the same as the defaults in the tf-agents.

| Configuration | Default Value | Discussion |
|---|---|---|
| **Proxy reward calculation**  |
| wirelength_weight | 1.0 |  |
| density_weight | 1.0 | Changed from 0.1 in the paper, to stabilize the training on Ariane.  |
| congestion_weight | 0.5 | Changed from 0.1 in the paper, to stabilize the training on Ariane. |
|---|---|---|
| **Standard cell placement**  |
| num_steps | [100, 100, 100] |  |
| io_factor | 1.0 |  |
| move_distance_factors | [1, 1, 1] |  |
| attract_factors | [100, 1e-3, 1e-5] |  |
| repel_factors | [0, 1e6, 1e7] |  |
|---|---|---|
| **Environment observation** |
| max_num_nodes | 4700 |  |
| max_num_edges | 28400 |  |
| max_grid_size | 128 |  |
| default_location_x | 0.5 |  |
| default_location_y | 0.5 |  |
|---|---|---|
| **Model architecture** |
| num_gcn_layers | 3 |  |
| edge_fc_layers | 1 |  |
| gcn_node_dim | 8 |  |
| dirichlet_alpha | 0.1 |  |
| policy_noise_weight | 0.0 |  |
|---|---|---|
| **Training** |
| optimizer | Adam | |
| learning_rate | 4e-4 | |
| sequence_length | 134 | |
| num_episodes_per_iteration | 1024 |  |
| global_batch_size | 1024 |  |
| num_epochs | 4 |  |
| value_pred_loss_coef | 0.5 | |
| entropy_regularization | 0.01 | |
| importance_ratio_clipping | 0.2 | |
| discount_factor | 1.0 | |
| entropy_regularization | 0.01 | |
| value_pred_loss_coef | 0.5 | |
| gradient_clipping | 1.0 | |
| use_gae | False | |
| use_td_lambda_return | False | |
| log_prob_clipping | 0.0 | |
| policy_l2_reg | 0.0 | |
| value_function_l2_reg | 0.0 | |
| shared_vars_l2_reg | 0.0 | |

<a id='Contributing'></a>
## How to contribute
We're eager to collaborate with you! See [`CONTRIBUTING.md`](CONTRIBUTING.md)
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

