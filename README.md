<!--* freshness: {
  owner: 'azalia'
  owner: 'tobyboyd'
  owner: 'sguada'
  owner: 'morpheus-oss-team'
  reviewed: '2021-09-01'
  review_interval: '12 months'
} *-->


# Circuit Training (Morpheus)

Open source of the Morpheus solution for use by the academic community and in
support of the paper published in Nature:
[A graph placement methodology for fast chip design](https://www.nature.com/articles/s41586-021-03544-w).

Disclaimer: This is not an official Google product.

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
