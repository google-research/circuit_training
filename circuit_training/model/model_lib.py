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
"""New circuittraining Model for generalization."""
import sys
from typing import Dict, Optional, Text, Union, Callable, Tuple

from circuit_training.environment import observation_config as observation_config_lib
import gin
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp


# Reimplements internal function
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/framework/smart_cond.py.
def smart_cond(pred: Union[bool, tf.Tensor],
               true_fn: Callable[[], tf.Tensor],
               false_fn: Callable[[], tf.Tensor],
               name: Optional[Text] = None) -> tf.Tensor:
  """Return either `true_fn()` if predicate `pred` is true else `false_fn()`.

  If `pred` is a bool or has a constant value, we return either `true_fn()`
  or `false_fn()`, otherwise we use `tf.cond` to dynamically route to both.

  Arguments:
    pred: A scalar determining whether to return the result of `true_fn` or
      `false_fn`.
    true_fn: The callable to be performed if pred is true.
    false_fn: The callable to be performed if pred is false.
    name: Optional name prefix when using `tf.cond`.

  Returns:
    Tensors returned by the call to either `true_fn` or `false_fn`.

  Raises:
    TypeError: If `true_fn` or `false_fn` is not callable.
  """
  if not callable(true_fn):
    raise TypeError('`true_fn` must be callable.')
  if not callable(false_fn):
    raise TypeError('`false_fn` must be callable.')
  pred_value = tf.get_static_value(pred)
  if isinstance(pred, tf.Tensor) or pred_value is None:
    return tf.cond(pred, true_fn=true_fn, false_fn=false_fn, name=name)
  if pred_value:
    return true_fn()
  else:
    return false_fn()


@gin.configurable
class CircuitTrainingModel(tf.keras.layers.Layer):
  """GCN-based model for circuit training."""

  EPSILON = 1E-6

  def __init__(
      self,
      static_features: Optional[Dict[Text, np.ndarray]] = None,
      observation_config: Optional[
          observation_config_lib.ObservationConfig] = None,
      num_gcn_layers: int = 3,
      edge_fc_layers: int = 1,
      gcn_node_dim: int = 8,
      dirichlet_alpha: float = 0.1,
      policy_noise_weight: float = 0.0,
      seed: int = 0,
  ):
    """Builds the circuit training model.

    Args:
      static_features: Optional static features that are invariant across steps
        on the same netlist, such as netlist metadata and the adj graphs. If not
        provided, use the input features in the call method.
      observation_config: Optional observation config.
      num_gcn_layers: Number of GCN layers.
      edge_fc_layers: Number of fully connected layers in the GCN kernel.
      gcn_node_dim: Node embedding dimension.
      dirichlet_alpha: Dirichlet concentration value.
      policy_noise_weight: Weight of the noise added to policy.
      seed: Seed for sampling noise.
    """
    super(CircuitTrainingModel, self).__init__()
    self._num_gcn_layers = num_gcn_layers
    self._gcn_node_dim = gcn_node_dim
    self._dirichlet_alpha = dirichlet_alpha
    self._policy_noise_weight = policy_noise_weight
    self._seed = seed
    self._static_features = static_features
    self._observation_config = (
        observation_config or observation_config_lib.ObservationConfig())

    seed = tfp.util.SeedStream(self._seed, salt='kernel_initializer_seed')
    kernel_initializer = tf.keras.initializers.glorot_uniform(seed=seed() %
                                                              sys.maxsize)

    self._metadata_encoder = tf.keras.Sequential([
        tf.keras.layers.Dense(
            self._gcn_node_dim, kernel_initializer=kernel_initializer),
        tf.keras.layers.ReLU(),
    ],
                                                 name='metadata_encoder')
    self._feature_encoder = tf.keras.Sequential([
        tf.keras.layers.Dense(
            self._gcn_node_dim, kernel_initializer=kernel_initializer),
        tf.keras.layers.ReLU(),
    ],
                                                name='feature_encoder')

    # Edge-centric GCN layers.
    def create_edge_fc(name=None) -> tf.keras.layers.Layer:
      seq = tf.keras.Sequential(name=name)
      for _ in range(edge_fc_layers):
        seq.add(
            tf.keras.layers.Dense(
                self._gcn_node_dim, kernel_initializer=kernel_initializer))
        seq.add(tf.keras.layers.ReLU())
      return seq

    self._edge_fc_list = [
        create_edge_fc(name='edge_fc_%d' % i)
        for i in range(self._num_gcn_layers)
    ]

    # Dot-product attention layer, a.k.a. Luong-style attention [1].
    # [1] Luong, et al, 2015.
    self._attention_layer = tf.keras.layers.Attention(name='attention_layer')
    self._attention_query_layer = tf.keras.layers.Dense(
        self._gcn_node_dim,
        name='attention_query_layer',
        kernel_initializer=kernel_initializer)
    self._attention_key_layer = tf.keras.layers.Dense(
        self._gcn_node_dim,
        name='attention_key_layer',
        kernel_initializer=kernel_initializer)
    self._attention_value_layer = tf.keras.layers.Dense(
        self._gcn_node_dim,
        name='attention_value_layer',
        kernel_initializer=kernel_initializer)

    self._value_head = tf.keras.Sequential([
        tf.keras.layers.Dense(32, kernel_initializer=kernel_initializer),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Dense(8, kernel_initializer=kernel_initializer),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Dense(1, kernel_initializer=kernel_initializer),
    ],
                                           name='value_head')

    # GAN-like deconv layers to generated the policy image.
    # See figures in http://shortn/_9HCSFwasnu.
    self._policy_location_head = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(
                (self._observation_config.max_grid_size // 16 *
                 self._observation_config.max_grid_size // 16 * 32),
                kernel_initializer=kernel_initializer),
            # 128/16*128/16*32 = 8*8*32
            tf.keras.layers.ReLU(),
            tf.keras.layers.Reshape(
                target_shape=(self._observation_config.max_grid_size // 16,
                              self._observation_config.max_grid_size // 16,
                              32)),
            # 8x8x32
            tf.keras.layers.Conv2DTranspose(
                filters=16,
                kernel_size=3,
                strides=2,
                padding='same',
                kernel_initializer=kernel_initializer),
            # 16x16x16
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2DTranspose(
                filters=8,
                kernel_size=3,
                strides=2,
                padding='same',
                kernel_initializer=kernel_initializer),
            # 32x32x8
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2DTranspose(
                filters=4,
                kernel_size=3,
                strides=2,
                padding='same',
                kernel_initializer=kernel_initializer),
            # 64x64x4
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2DTranspose(
                filters=2,
                kernel_size=3,
                strides=2,
                padding='same',
                kernel_initializer=kernel_initializer),
            # 128x128x2
            tf.keras.layers.ReLU(),
            # No activation.
            tf.keras.layers.Conv2DTranspose(
                filters=1,
                kernel_size=3,
                strides=1,
                padding='same',
                kernel_initializer=kernel_initializer),
            # 128x128x1
            tf.keras.layers.Flatten()
        ],
        name='policy_location_head')

  def _scatter_count(self, edge_h: tf.Tensor,
                     indices: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    """Aggregate (reduce sum) edge embeddings to nodes.

    Args:
      edge_h: A [-1, #edges, h] tensor of edge embeddings.
      indices: A [-1, #edges] tensor of node index of an edge (sparse adjacency
        indices).

    Returns:
      A [-1, #nodes, h] tensor of aggregated node embeddings and a
      [-1, #nodes, 1] tensor of edge count per node for finding the mean.
    """
    batch = tf.shape(edge_h)[0]
    num_items = tf.shape(edge_h)[1]
    num_lattents = edge_h.shape[2]

    h_node = tf.zeros(
        [batch, self._observation_config.max_num_nodes, num_lattents])
    count_edge = tf.zeros_like(h_node)
    count = tf.ones_like(edge_h)

    b_indices = tf.tile(
        tf.expand_dims(tf.range(0, tf.cast(batch, dtype=tf.int32)), -1),
        [1, num_items])
    idx = tf.stack([b_indices, indices], axis=-1)
    h_node = tf.tensor_scatter_nd_add(h_node, idx, edge_h)
    count_edge = tf.tensor_scatter_nd_add(count_edge, idx, count)

    return h_node, count_edge

  def gather_to_edges(
      self, h_nodes: tf.Tensor, sparse_adj_i: tf.Tensor,
      sparse_adj_j: tf.Tensor,
      sparse_adj_weight: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    """Gathers node embeddings to edges.

    For each edge, there are two node embeddings. It concats them together with
    the edge weight. It also masks the output with 0 for edges with no weight.

    Args:
      h_nodes: A [-1, #node, h] tensor of node embeddings.
      sparse_adj_i: A [-1, #edges] tensor for the 1st node index of an edge.
      sparse_adj_j: A [-1, #edges] tensor for the 2nd node index of an edge.
      sparse_adj_weight: A [-1, #edges] tensor for the weight of an edge. 0 for
        fake padded edges.

    Returns:
      A [-1, #edges, 2*h+1], [-1, #edges, 2*h+1]  tensor of edge embeddings.
    """

    h_edges_1 = tf.gather(h_nodes, sparse_adj_i, batch_dims=1)
    h_edges_2 = tf.gather(h_nodes, sparse_adj_j, batch_dims=1)
    h_edges_12 = tf.concat([h_edges_1, h_edges_2, sparse_adj_weight], axis=-1)
    h_edges_21 = tf.concat([h_edges_2, h_edges_1, sparse_adj_weight], axis=-1)
    mask = tf.broadcast_to(
        tf.not_equal(sparse_adj_weight, 0.0), tf.shape(h_edges_12))
    h_edges_i_j = tf.where(mask, h_edges_12, tf.zeros_like(h_edges_12))
    h_edges_j_i = tf.where(mask, h_edges_21, tf.zeros_like(h_edges_21))
    return h_edges_i_j, h_edges_j_i

  def scatter_to_nodes(self, h_edges: tf.Tensor, sparse_adj_i: tf.Tensor,
                       sparse_adj_j: tf.Tensor) -> tf.Tensor:
    """Scatters edge embeddings to nodes via mean aggregation.

    For each node, it aggregates the embeddings of all the connected edges by
    averaging them.

    Args:
      h_edges: A [-1, #edges, h] tensor of edge embeddings.
      sparse_adj_i: A [-1, #edges] tensor for the 1st node index of an edge.
      sparse_adj_j: A [-1, #edges] tensor for the 2nd node index of an edge.

    Returns:
      A [-1, #node, h] tensor of node embeddings.
    """
    h_nodes_1, count_1 = self._scatter_count(h_edges, sparse_adj_i)
    h_nodes_2, count_2 = self._scatter_count(h_edges, sparse_adj_j)
    return (h_nodes_1 + h_nodes_2) / (count_1 + count_2 + self.EPSILON)

  def self_attention(self,
                     h_current_node: tf.Tensor,
                     h_nodes: tf.Tensor,
                     training: bool = False) -> tf.Tensor:
    """Returns self-attention wrt to the current node.

    Args:
      h_current_node: A [-1, 1, h] tensor of the current node embedding.
      h_nodes: A [-1, #nodes, h] tensor of all node embeddings.
      training: Set in the training mode.

    Returns:
      A [-1, h] tensor of the weighted average of the node embeddings where
      the weight is the attention score with respect to the current node.
    """
    query = self._attention_query_layer(h_current_node, training=training)
    values = self._attention_value_layer(h_nodes, training=training)
    keys = self._attention_key_layer(h_nodes, training=training)
    h_attended = self._attention_layer([query, values, keys], training=training)
    h_attended = tf.squeeze(h_attended, axis=1)
    return h_attended

  def add_noise(self, logits: tf.Tensor) -> tf.Tensor:
    """Adds a non-trainable dirichlet noise to the policy."""
    seed = tfp.util.SeedStream(self._seed, salt='noise_seed')

    probs = tf.nn.softmax(logits)
    alphas = tf.fill(tf.shape(probs), self._dirichlet_alpha)
    dirichlet_distribution = tfp.distributions.Dirichlet(alphas)
    noise = dirichlet_distribution.sample(seed=seed() % sys.maxsize)
    noised_probs = ((1.0 - self._policy_noise_weight) * probs +
                    (self._policy_noise_weight) * noise)

    noised_logit = tf.math.log(noised_probs + self.EPSILON)

    return noised_logit

  def _get_static_input(self, static_feature_key: Text,
                        inputs: Dict[Text, tf.Tensor]) -> tf.Tensor:
    """Returns the tensor for a particular static feature.

    Args:
      static_feature_key: a feature key defined in
        observation_config_lib.STATIC_OBSERVATIONS
      inputs: the dictionary of input features.

    Returns:
      A tensor for the static feature.
    """
    if self._static_features:
      # For the online single-netlist training, replicates feature by batch
      # size. Picking an aribitrary non-static feature to get a reference of
      # the dynamic dimension at runtime.
      num_batches_dim = tf.shape(inputs['current_node'])[0]
      return tf.tile(
          tf.expand_dims(self._static_features[static_feature_key], 0),
          [num_batches_dim, 1])
    else:
      # For the offline multi-netlist training, reading the static feature from
      # the inputs.
      return inputs[static_feature_key]

  def call(self,
           inputs: tf.Tensor,
           training: bool = False,
           is_eval: bool = False) -> Tuple[Dict[Text, tf.Tensor], tf.Tensor]:
    # Netlist metadata.
    netlist_metadata_inputs = [
        self._get_static_input(key, inputs)
        for key in observation_config_lib.NETLIST_METADATA
    ]

    # Graph.
    # pytype: disable=wrong-arg-types  # dynamic-method-lookup
    sparse_adj_weight = self._get_static_input('sparse_adj_weight', inputs)
    sparse_adj_i = tf.cast(
        self._get_static_input('sparse_adj_i', inputs), dtype=tf.int32)
    sparse_adj_j = tf.cast(
        self._get_static_input('sparse_adj_j', inputs), dtype=tf.int32)

    # Node features.
    node_types = self._get_static_input('node_types', inputs)
    is_node_placed = tf.cast(inputs['is_node_placed'], dtype=tf.float32)
    macros_w = self._get_static_input('macros_w', inputs)
    macros_h = self._get_static_input('macros_h', inputs)
    # pytype: enable=wrong-arg-types  # dynamic-method-lookup
    locations_x = inputs['locations_x']
    locations_y = inputs['locations_y']

    # Current node.
    current_node = tf.cast(inputs['current_node'], dtype=tf.int32)

    is_hard_macro = tf.cast(
        tf.math.equal(node_types, observation_config_lib.HARD_MACRO),
        dtype=tf.float32)
    is_soft_macro = tf.cast(
        tf.math.equal(node_types, observation_config_lib.SOFT_MACRO),
        dtype=tf.float32)
    is_port_cluster = tf.cast(
        tf.math.equal(node_types, observation_config_lib.PORT_CLUSTER),
        dtype=tf.float32)

    netlist_metadata = tf.concat(netlist_metadata_inputs, axis=1)
    h_metadata = self._metadata_encoder(netlist_metadata, training=training)

    h_nodes = tf.stack([
        locations_x,
        locations_y,
        macros_w,
        macros_h,
        is_hard_macro,
        is_soft_macro,
        is_port_cluster,
        is_node_placed,
    ],
                       axis=2)

    h_nodes = self._feature_encoder(h_nodes, training=training)

    # Edge-centric GCN
    #
    # Here, we are using a modified version of Graph Convolutional Network
    # (GCN)[1] that focuses on edge properties rather than node properties.
    # In this modified GCN, the features of neighbouring nodes are
    # mixed together to create edge features. Then, edge features are
    # aggregated on the connected nodes to create the output node embedding.
    # The GCN message passing happens indirectly between neighbouring nodes
    # through the mixing on the edges.
    #
    # Edge-centric GCN for Circuit Training
    #
    # The nodes of the circuit training observation graph are hard macros,
    # soft macros, and port clusters and the edges are the wires between them.
    # The intuition behind using edge-centric GCN was that the wirelength and
    # congestion costs (reward signals) depends on properties of the
    # wires (edge) and not the macros.
    # This architecture has shown promising results on predicting supervised
    # graph regression for predicting wirelength and congestion and we hope
    # it performs well in reinforcement setting to predict value and policy.
    #
    # An alternative approach was applying original GCN on the Line Graph of
    # the ckt graph (see https://en.wikipedia.org/wiki/Line_graph).
    # Nodes of the line graph correspond to the edges of the original graph.
    # However, the adjacency matrix of the line graph will be prohibitively
    # large and can't be readily processed by GCN.
    #
    # See figures in http://shortn/_j1NsgZBqAr for edge-centric GCN.
    #
    # [1] Kipf and Welling, 2016.
    sparse_adj_weight = tf.expand_dims(
        sparse_adj_weight, axis=-1, name='sparse_adj_weight')

    for i in range(self._num_gcn_layers):
      # For bi-directional graph.
      h_edges_i_j, h_edges_j_i = self.gather_to_edges(
          h_nodes=h_nodes,
          sparse_adj_i=sparse_adj_i,
          sparse_adj_j=sparse_adj_j,
          sparse_adj_weight=sparse_adj_weight)
      h_edges = (self._edge_fc_list[i](h_edges_i_j, training=training) +
                 self._edge_fc_list[i](h_edges_j_i, training=training)) / 2.0
      h_nodes_new = self.scatter_to_nodes(h_edges, sparse_adj_i, sparse_adj_j)
      # Skip connection.
      h_nodes = h_nodes_new + h_nodes

    observation_hiddens = []
    observation_hiddens.append(h_metadata)

    h_all_edges = tf.reduce_mean(h_edges, axis=1)
    observation_hiddens.append(h_all_edges)

    h_current_node = tf.gather(h_nodes, current_node, batch_dims=1)

    h_attended = self.self_attention(h_current_node, h_nodes, training=training)
    observation_hiddens.append(h_attended)

    h_current_node = tf.squeeze(h_current_node, axis=1)
    observation_hiddens.append(h_current_node)

    h = tf.concat(observation_hiddens, axis=1)

    location_logits = self._policy_location_head(h, training=training)
    # smart_cond avoids using tf.cond when condition value is static.
    logits = {
        'location':
            smart_cond(is_eval, lambda: location_logits,
                       lambda: self.add_noise(location_logits)),
    }
    value = self._value_head(h, training=training)

    return logits, value


class CircuitTrainingTPUModel(CircuitTrainingModel):
  """Model optimized for TPU performance using map_fn."""

  def _scatter_count(self, edge_h: tf.Tensor,
                     indices: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    """Aggregate (reduce sum) edge embeddings to nodes.

    Args:
      edge_h: A [-1, #edges, h] tensor of edge embeddings.
      indices: A [-1, #edges] tensor of node index of an edge (sparse adjacency
        indices).

    Returns:
      A [-1, #nodes, h] tensor of aggregated node embeddings and a
      [-1, #nodes, 1] tensor of edge count per node for finding the mean.
    """
    batch = tf.shape(edge_h)[0]
    num_lattents = edge_h.shape[2]
    h_node = tf.zeros(
        [batch, self._observation_config.max_num_nodes, num_lattents])
    count_edge = tf.zeros_like(h_node)
    count = tf.ones_like(edge_h)

    indices = tf.expand_dims(indices, axis=-1)
    h_node = tf.map_fn(
        lambda x: tf.tensor_scatter_nd_add(x[0], x[1], x[2]),
        (h_node, indices, edge_h),
        fn_output_signature=tf.TensorSpec(
            shape=(self._observation_config.max_num_nodes, num_lattents)))
    count_edge = tf.map_fn(
        lambda x: tf.tensor_scatter_nd_add(x[0], x[1], x[2]),
        (count_edge, indices, count),
        fn_output_signature=tf.TensorSpec(
            shape=(self._observation_config.max_num_nodes, num_lattents)))
    return h_node, count_edge

  def gather_to_edges(
      self, h_nodes: tf.Tensor, sparse_adj_i: tf.Tensor,
      sparse_adj_j: tf.Tensor,
      sparse_adj_weight: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    """Gathers node embeddings to edges.

    For each edge, there are two node embeddings. It concats them together with
    the edge weight. It also masks the output with 0 for edges with no weight.

    Args:
      h_nodes: A [-1, #node, h] tensor of node embeddings.
      sparse_adj_i: A [-1, #edges] tensor for the 1st node index of an edge.
      sparse_adj_j: A [-1, #edges] tensor for the 2nd node index of an edge.
      sparse_adj_weight: A [-1, #edges] tensor for the weight of an edge. 0 for
        fake padded edges.

    Returns:
      A [-1, #edges, 2*h+1], [-1, #edges, 2*h+1]  tensor of edge embeddings.
    """

    h_edges_1 = tf.map_fn(
        lambda x: tf.gather(x[0], x[1], batch_dims=0),
        (h_nodes, sparse_adj_i),
        fn_output_signature=tf.float32)
    h_edges_2 = tf.map_fn(
        lambda x: tf.gather(x[0], x[1], batch_dims=0),
        (h_nodes, sparse_adj_j),
        fn_output_signature=tf.float32)
    h_edges = tf.concat([h_edges_1, h_edges_2, sparse_adj_weight], axis=2)
    mask = tf.broadcast_to(sparse_adj_weight != 0.0, tf.shape(h_edges))
    return tf.where(mask, h_edges, tf.zeros_like(h_edges))

  def call(self,
           inputs: tf.Tensor,
           training: bool = False,
           is_eval: bool = False) -> Tuple[Dict[Text, tf.Tensor], tf.Tensor]:
    # Netlist metadata.
    netlist_metadata_inputs = [
        self._get_static_input(key, inputs)
        for key in observation_config_lib.NETLIST_METADATA
    ]

    # Graph.
    sparse_adj_weight = self._get_static_input('sparse_adj_weight', inputs)
    sparse_adj_i = tf.cast(
        self._get_static_input('sparse_adj_i', inputs), dtype=tf.int32)
    sparse_adj_j = tf.cast(
        self._get_static_input('sparse_adj_j', inputs), dtype=tf.int32)

    # Node features.
    node_types = self._get_static_input('node_types', inputs)
    is_node_placed = tf.cast(inputs['is_node_placed'], dtype=tf.float32)
    macros_w = self._get_static_input('macros_w', inputs)
    macros_h = self._get_static_input('macros_h', inputs)
    locations_x = inputs['locations_x']
    locations_y = inputs['locations_y']

    # Current node.
    current_node = tf.cast(inputs['current_node'], dtype=tf.int32)

    is_hard_macro = tf.cast(
        tf.math.equal(node_types, observation_config_lib.HARD_MACRO),
        dtype=tf.float32)
    is_soft_macro = tf.cast(
        tf.math.equal(node_types, observation_config_lib.SOFT_MACRO),
        dtype=tf.float32)
    is_port_cluster = tf.cast(
        tf.math.equal(node_types, observation_config_lib.PORT_CLUSTER),
        dtype=tf.float32)

    netlist_metadata = tf.concat(netlist_metadata_inputs, axis=1)
    h_metadata = self._metadata_encoder(netlist_metadata, training=training)

    h_nodes = tf.stack([
        locations_x,
        locations_y,
        macros_w,
        macros_h,
        is_hard_macro,
        is_soft_macro,
        is_port_cluster,
        is_node_placed,
    ],
                       axis=2)

    h_nodes = self._feature_encoder(h_nodes, training=training)

    # Edge-centric GCN
    #
    # Here, we are using a modified version of Graph Convolutional Network
    # (GCN)[1] that focuses on edge properties rather than node properties.
    # In this modified GCN, the features of neighbouring nodes are
    # mixed together to create edge features. Then, edge features are
    # aggregated on the connected nodes to create the output node embedding.
    # The GCN message passing happens indirectly between neighbouring nodes
    # through the mixing on the edges.
    #
    # Edge-centric GCN for Circuit Training
    #
    # The nodes of the circuit training observation graph are hard macros,
    # soft macros, and port clusters and the edges are the wires between them.
    # The intuition behind using edge-centric GCN was that the wirelength and
    # congestion costs (reward signals) depends on properties of the
    # wires (edge) and not the macros.
    # This architecture has shown promising results on predicting supervised
    # graph regression for predicting wirelength and congestion and we hope
    # it performs well in reinforcement setting to predict value and policy.
    #
    # An alternative approach was applying original GCN on the Line Graph of
    # the ckt graph (see https://en.wikipedia.org/wiki/Line_graph).
    # Nodes of the line graph correspond to the edges of the original graph.
    # However, the adjacency matrix of the line graph will be prohibitively
    # large and can't be readily processed by GCN.
    #
    # See figures in http://shortn/_j1NsgZBqAr for edge-centric GCN.
    #
    # [1] Kipf and Welling, 2016.

    def update_tpu(h_nodes, i=0):
      # Optimizing scatter/gather performance on TPUs.
      # For bi-directional graph.
      h_edges_1 = tf.map_fn(
          lambda x: tf.gather(x[0], x[1], batch_dims=0),
          (h_nodes, sparse_adj_i),
          fn_output_signature=tf.float32)
      h_edges_2 = tf.map_fn(
          lambda x: tf.gather(x[0], x[1], batch_dims=0),
          (h_nodes, sparse_adj_j),
          fn_output_signature=tf.float32)

      h_edges_12 = tf.concat([h_edges_1, h_edges_2, sparse_adj_weight], axis=-1)
      mask = tf.broadcast_to(sparse_adj_weight != 0.0, tf.shape(h_edges_12))
      h_edges_i_j = tf.where(mask, h_edges_12, tf.zeros_like(h_edges_12))

      # h_edges_j_i = self.gather_to_edges(
      h_edges_21 = tf.concat([h_edges_2, h_edges_1, sparse_adj_weight], axis=-1)
      h_edges_j_i = tf.where(mask, h_edges_21, tf.zeros_like(h_edges_21))

      h_edges = (self._edge_fc_list[i](h_edges_i_j, training=training) +
                 self._edge_fc_list[i](h_edges_j_i, training=training)) / 2.0

      h_node = tf.zeros_like(h_nodes)
      num_lattents = h_edges.shape[2]
      count_edge = tf.zeros_like(h_node)
      count = tf.ones_like(h_edges)
      indices = tf.expand_dims(sparse_adj_i, axis=-1)
      h_nodes_1 = tf.map_fn(
          lambda x: tf.tensor_scatter_nd_add(x[0], x[1], x[2]),
          (h_node, indices, h_edges),
          fn_output_signature=tf.TensorSpec(
              shape=(self._observation_config.max_num_nodes, num_lattents)))
      count_1 = tf.map_fn(
          lambda x: tf.tensor_scatter_nd_add(x[0], x[1], x[2]),
          (count_edge, indices, count),
          fn_output_signature=tf.TensorSpec(
              shape=(self._observation_config.max_num_nodes, num_lattents)))
      indices = tf.expand_dims(sparse_adj_j, axis=-1)
      h_nodes_2 = tf.map_fn(
          lambda x: tf.tensor_scatter_nd_add(x[0], x[1], x[2]),
          (h_node, indices, h_edges),
          fn_output_signature=tf.TensorSpec(
              shape=(self._observation_config.max_num_nodes, num_lattents)))
      count_2 = tf.map_fn(
          lambda x: tf.tensor_scatter_nd_add(x[0], x[1], x[2]),
          (count_edge, indices, count),
          fn_output_signature=tf.TensorSpec(
              shape=(self._observation_config.max_num_nodes, num_lattents)))

      h_nodes_new = (h_nodes_1 + h_nodes_2) / (count_1 + count_2 + self.EPSILON)
      # Skip connection.
      h_nodes = h_nodes_new + h_nodes
      return h_nodes, h_edges

    sparse_adj_weight = tf.expand_dims(
        sparse_adj_weight, axis=-1, name='sparse_adj_weight')

    h_nodes = tf.identity(h_nodes, 'initial_h_nodes')
    for i in range(self._num_gcn_layers):
      h_nodes, h_edges = update_tpu(h_nodes, i)

    observation_hiddens = []
    observation_hiddens.append(h_metadata)

    h_all_edges = tf.reduce_mean(h_edges, axis=1)
    observation_hiddens.append(h_all_edges)

    h_current_node = tf.map_fn(
        lambda x: tf.gather(x[0], x[1], batch_dims=0),
        (h_nodes, current_node),
        fn_output_signature=tf.float32)

    h_attended = self.self_attention(h_current_node, h_nodes, training=training)
    observation_hiddens.append(h_attended)

    h_current_node = tf.squeeze(h_current_node, axis=1)
    observation_hiddens.append(h_current_node)

    h = tf.concat(observation_hiddens, axis=1)

    location_logits = self._policy_location_head(h, training=training)
    # smart_cond avoids using tf.cond when condition value is static.
    logits = {
        'location':
            smart_cond(is_eval, lambda: location_logits,
                       lambda: self.add_noise(location_logits)),
    }
    value = self._value_head(h, training=training)

    return logits, value
