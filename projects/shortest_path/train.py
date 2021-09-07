# Copyright 2021 Petrov, Danil <ddbihbka@gmail.com>
# Author: Petrov, Danil <ddbihbka@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Callable, Sequence

import functools
import time

from absl import logging
from clu import metric_writers
from clu import periodic_actions

import jax
import jax.numpy as jnp

from jax import lax

from flax import jax_utils
from flax import optim
from flax import linen as nn

from flax.training import train_state
from flax.training import checkpoints
from flax.training import common_utils

import jraph
import optax
import ml_collections

import tensorflow as tf
import tensorflow_datasets.public_api as tfds

import projects.shortest_path.input_pipeline as input_pipeline
# pylint: disable=unused-import
import datasets.graphs.shortest_path
# pylint: enable=unused-import


class ExplicitMLP(nn.Module):
    """A flax MLP."""
    features: Sequence[int]

    @nn.compact
    def __call__(self, inputs):
        x = inputs
        for i, lyr in enumerate([nn.Dense(feat) for feat in self.features]):
            x = lyr(x)
            x = nn.LayerNorm()(x)
            if i != len(self.features) - 1:
                x = nn.swish(x)
        return x


# Functions must be passed to jraph GNNs, but pytype does not recognise
# linen Modules as callables to here we wrap in a function.
def make_embed_fn(latent_size, norm=True):

    def embed(inputs):
        x = nn.Dense(latent_size)(inputs)
        if norm:
            x = nn.LayerNorm()(x)
        return x

    return embed


def make_mlp(features):

    @jraph.concatenated_args
    def update_fn(inputs):
        return ExplicitMLP(features)(inputs)

    return update_fn


class GraphNetwork(nn.Module):
    """A flax GraphNetwork."""
    mlp_features: Sequence[int]
    latent_size: int
    message_passing_num: int

    @nn.compact
    def __call__(self, graph):

        encoder = jraph.GraphMapFeatures(
            embed_node_fn=make_embed_fn(self.latent_size),
            embed_edge_fn=make_embed_fn(self.latent_size),
            embed_global_fn=make_embed_fn(self.latent_size))
        core = jraph.GraphNetwork(
            update_node_fn=make_mlp(self.mlp_features),
            update_edge_fn=make_mlp(self.mlp_features),
        # The global update outputs size 2 for binary classification.
            update_global_fn=make_mlp(self.mlp_features))    # pytype: disable=unsupported-operands
        decoder = jraph.GraphMapFeatures(
            embed_node_fn=make_embed_fn(self.latent_size),
            embed_edge_fn=make_embed_fn(self.latent_size),
            embed_global_fn=make_embed_fn(self.latent_size))

        output_transform = jraph.GraphMapFeatures(
            embed_node_fn=make_embed_fn(2, norm=False),
            embed_edge_fn=make_embed_fn(2, norm=False),
            embed_global_fn=make_embed_fn(1, norm=False))

        graph = encoder(graph)

        for _ in range(self.message_passing_num):
            graph = decoder(core(graph))

        graph = output_transform(graph)
        return graph


def initialized(key, model: nn.Module):
    n_node = 13
    n_edge = n_node * n_node
    senders, receivers = jnp.indices((n_node, n_node))

    @jax.jit
    def init(*args):
        return model.init(*args)

    variables = init({'params': key},
                     jraph.GraphsTuple(
                         n_node=jnp.array([n_node]),
                         n_edge=jnp.array([n_edge]),
                         nodes=jnp.ones((n_node, 5), jnp.float32),
                         edges=jnp.ones((n_edge, 1), jnp.float32),
                         globals=jnp.zeros((1, 1)),
                         senders=jnp.ravel(senders),
                         receivers=jnp.ravel(receivers),
                     ))
    return variables['params']


class TrainState(train_state.TrainState):
    dynamic_scale: optim.DynamicScale


def create_train_state(rng, config: ml_collections.ConfigDict, model: nn.Module,
                       learning_rate_fn: Callable):
    """Create initial training state."""
    dynamic_scale = None
    platform = jax.local_devices()[0].platform
    if config.half_precision and platform == 'gpu':
        dynamic_scale = optim.DynamicScale()
    else:
        dynamic_scale = None

    params = initialized(rng, model)
    tx = optax.adam(learning_rate=learning_rate_fn,)
    state = TrainState.create(apply_fn=model.apply,
                              params=params,
                              tx=tx,
                              dynamic_scale=dynamic_scale)
    return state


def restore_checkpoint(state, workdir):
    return checkpoints.restore_checkpoint(workdir, state)


def save_checkpoint(state, workdir):
    if jax.process_index() == 0:
        # get train state from the first replica
        state = jax.device_get(jax.tree_map(lambda x: x[0], state))
        step = int(state.step)
        checkpoints.save_checkpoint(workdir, state, step, keep=3)


def create_learning_rate_fn(config: ml_collections.ConfigDict,
                            base_learning_rate: float,
                            steps_per_epoch: int) -> Callable:
    """Create learning rate schedule."""
    warmup_fn = optax.linear_schedule(init_value=0.,
                                      end_value=base_learning_rate,
                                      transition_steps=config.warmup_epochs *
                                      steps_per_epoch)
    cosine_epochs = max(config.num_epochs - config.warmup_epochs, 1)
    cosine_fn = optax.cosine_decay_schedule(init_value=base_learning_rate,
                                            decay_steps=cosine_epochs *
                                            steps_per_epoch)
    schedule_fn = optax.join_schedules(
        schedules=[warmup_fn, cosine_fn],
        boundaries=[config.warmup_epochs * steps_per_epoch])
    return schedule_fn


def cross_entropy_loss(logits, labels):
    one_hot_labels = common_utils.onehot(labels, num_classes=2)
    xentropy = optax.softmax_cross_entropy(logits=logits, labels=one_hot_labels)
    return jnp.mean(xentropy)


def compute_metrics(logits, node_labels, edge_labels):
    loss = cross_entropy_loss(logits.nodes, node_labels) + cross_entropy_loss(
        logits.edges, edge_labels)

    accuracy = (jnp.mean(jnp.argmax(logits.nodes, -1) == node_labels) +
                jnp.mean(jnp.argmax(logits.nodes, -1) == node_labels)) / 2.0
    metrics = {
        'loss': loss,
        'accuracy': accuracy,
    }
    metrics = lax.pmean(metrics, axis_name='batch')
    return metrics


def train_step(state, batch, learning_rate_fn):
    """Perform a single training step."""

    def loss_fn(params):
        """loss function used for training."""
        preds = state.apply_fn({'params': params}, batch.graph)
        loss = (cross_entropy_loss(preds.nodes, batch.node_labels) +
                cross_entropy_loss(preds.edges, batch.edge_labels))
        weight_penalty_params = jax.tree_leaves(params)
        weight_decay = 0.0001
        weight_l2 = sum(
            [jnp.sum(x**2) for x in weight_penalty_params if x.ndim > 1])
        weight_penalty = weight_decay * 0.5 * weight_l2
        loss = loss + weight_penalty
        return loss, preds

    step = state.step
    dynamic_scale = state.dynamic_scale
    lr = learning_rate_fn(step)

    if dynamic_scale:
        grad_fn = dynamic_scale.value_and_grad(loss_fn,
                                               has_aux=True,
                                               axis_name='batch')
        dynamic_scale, is_fin, aux, grads = grad_fn(state.params)
        # dynamic loss takes care of averaging gradients across replicas
    else:
        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        aux, grads = grad_fn(state.params)
        # Re-use same axis_name as in the call to `pmap(...train_step...)` below.
        grads = lax.pmean(grads, axis_name='batch')
    logits = aux[1]
    metrics = compute_metrics(logits, batch.node_labels, batch.edge_labels)
    metrics['learning_rate'] = lr

    new_state = state.apply_gradients(grads=grads)
    if dynamic_scale:
        # if is_fin == False the gradients contain Inf/NaNs and optimizer state and
        # params should be restored (= skip this step).
        new_state = new_state.replace(
            opt_state=jax.tree_multimap(functools.partial(jnp.where, is_fin),
                                        new_state.opt_state, state.opt_state),
            params=jax.tree_multimap(functools.partial(jnp.where, is_fin),
                                     new_state.params, state.params))
        metrics['scale'] = dynamic_scale.scale

    return new_state, metrics


def eval_step(state, batch):
    variables = {'params': state.params}
    preds = state.apply_fn(variables, batch.graph, mutable=False)
    return compute_metrics(preds, batch.node_labels, batch.edge_labels)


def train_and_evaluate(config: ml_collections.ConfigDict,
                       workdir: str) -> TrainState:
    """Execute model training and evaluation loop.
      Args:
        config: Hyperparameter configuration for training and evaluation.
        workdir: Directory where the tensorboard summaries are written to.
      Returns:
        Final TrainState.
    """
    writer = metric_writers.create_default_writer(
        logdir=workdir, just_logging=jax.host_id() != 0)

    rng = jax.random.PRNGKey(0)

    if config.batch_size % jax.device_count() > 0:
        raise ValueError(
            'Batch size must be divisible by the number of devices')
    local_batch_size = config.batch_size // jax.process_count()

    platform = jax.local_devices()[0].platform

    if config.half_precision:
        if platform == 'tpu':
            input_dtype = tf.bfloat16
        else:
            input_dtype = tf.float16
    else:
        input_dtype = tf.float32

    dataset_builder = tfds.builder(config.dataset)
    dataset_builder.download_and_prepare()
    train_iter = jax_utils.prefetch_to_device(
        input_pipeline.create_input_iter(dataset_builder,
                                         local_batch_size,
                                         config.n_node_per_graph,
                                         config.n_edge_per_graph,
                                         input_dtype,
                                         train=True,
                                         cache=config.cache), 10)
    eval_iter = jax_utils.prefetch_to_device(
        input_pipeline.create_input_iter(dataset_builder,
                                         local_batch_size,
                                         config.n_node_per_graph,
                                         config.n_edge_per_graph,
                                         input_dtype,
                                         train=False,
                                         cache=config.cache), 10)

    steps_per_epoch = (dataset_builder.info.splits['train'].num_examples //
                       config.batch_size)

    if config.num_train_steps == -1:
        num_steps = int(steps_per_epoch * config.num_epochs)
    else:
        num_steps = config.num_train_steps

    if config.steps_per_eval == -1:
        num_validation_examples = dataset_builder.info.splits[
            'test'].num_examples
        steps_per_eval = num_validation_examples // config.batch_size
    else:
        steps_per_eval = config.steps_per_eval

    steps_per_checkpoint = steps_per_epoch * 10

    base_learning_rate = config.learning_rate * config.batch_size / 256.

    model = GraphNetwork(mlp_features=(16,),
                         latent_size=16,
                         message_passing_num=3)

    learning_rate_fn = create_learning_rate_fn(config, base_learning_rate,
                                               steps_per_epoch)

    state = create_train_state(rng, config, model, learning_rate_fn)
    state = restore_checkpoint(state, workdir)
    # step_offset > 0 if restarting from checkpoint
    step_offset = int(state.step)
    state = jax_utils.replicate(state)

    p_train_step = jax.pmap(functools.partial(
        train_step, learning_rate_fn=learning_rate_fn),
                            axis_name='batch')
    p_eval_step = jax.pmap(eval_step, axis_name='batch')

    train_metrics = []
    hooks = []
    if jax.process_index() == 0:
        hooks += [periodic_actions.Profile(num_profile_steps=5, logdir=workdir)]
    train_metrics_last_t = time.time()
    logging.info('Initial compilation, this might take some minutes...')
    for step, batch in zip(range(step_offset, num_steps), train_iter):
        state, metrics = p_train_step(state, batch)
        for h in hooks:
            h(step)
        if step == step_offset:
            logging.info('Initial compilation completed.')

        if config.get('log_every_steps'):
            train_metrics.append(metrics)
            if (step + 1) % config.log_every_steps == 0:
                train_metrics = common_utils.get_metrics(train_metrics)
                summary = {
                    f'train_{k}': v for k, v in jax.tree_map(
                        lambda x: x.mean(), train_metrics).items()
                }
                summary['steps_per_second'] = config.log_every_steps / (
                    time.time() - train_metrics_last_t)
                writer.write_scalars(step + 1, summary)
                train_metrics = []
                train_metrics_last_t = time.time()

        if (step + 1) % steps_per_epoch == 0:
            epoch = step // steps_per_epoch
            eval_metrics = []

            for _ in range(steps_per_eval):
                eval_batch = next(eval_iter)
                metrics = p_eval_step(state, eval_batch)
                eval_metrics.append(metrics)
            eval_metrics = common_utils.get_metrics(eval_metrics)
            summary = jax.tree_map(lambda x: x.mean(), eval_metrics)
            logging.info('eval epoch: %d, loss: %.4f, accuracy: %.2f', epoch,
                         summary['loss'], summary['accuracy'] * 100)
            writer.write_scalars(
                step + 1, {f'eval_{key}': val for key, val in summary.items()})
            writer.flush()
        if (step + 1) % steps_per_checkpoint == 0 or step + 1 == num_steps:
            save_checkpoint(state, workdir)

    # Wait until computations are done before exiting
    jax.random.normal(jax.random.PRNGKey(0), ()).block_until_ready()

    return state
