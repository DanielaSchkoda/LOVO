import os
from typing import Any, Dict, Tuple, Callable, Iterable

import optax
from flax.metrics.tensorboard import SummaryWriter
from flax.training.train_state import TrainState
from flax.training import checkpoints
from optax._src.wrappers import MultiSteps

from LOVO_via_DL.LOVO_DL_vs_baseline_vs_LiNGAM import infer_bivariate_structure
from jax_model.transitive_predictor import TransitivePredictor
import jax
from tqdm import tqdm
import jax.numpy as jnp


def get_default_log_dir() -> str:
    log_dir = 'jax_structure'
    run_number = 1

    # Find the next available run number
    while os.path.exists(os.path.join(log_dir, f"run_{run_number}")):
        run_number += 1
    print('Run {}'.format(run_number))
    log_dir = os.path.join(log_dir, f"run_{run_number}")
    return os.path.abspath(log_dir)


def get_optimiser(lr: float = 1e-4, grad_clip_value: float = 1, accumulate_grads: int = 1) -> MultiSteps:
    return optax.MultiSteps(optax.chain(
        optax.clip_by_global_norm(grad_clip_value),
        optax.adam(lr, b1=0.9, b2=0.99),
    ), accumulate_grads)


def ce_loss(output: jax.Array, target: jax.Array) -> jax.Array:
    print(output.shape, target.shape)
    jax.debug.print('{} {}', output.shape, target.shape)
    return optax.softmax_cross_entropy(logits=output, labels=target).mean()


@jax.jit
def train_step(state: TrainState, rng, batch):

    def forward_pass(params, rng, batch) -> Tuple:
        xz_data, z_target = batch
        rng, dropout_apply_rng = jax.random.split(rng)
        output = state.apply_fn({'params': params}, xz_data,
                                train=True,
                                rngs={'dropout': dropout_apply_rng}
                                )
        mse = ce_loss(output, z_target)
        return mse, (output, rng)
    # Gradient function
    grad_fn = jax.value_and_grad(forward_pass, has_aux=True)
    # Determine gradients for current model, parameters and batch
    (loss, (output, rng)), grads = grad_fn(state.params, rng, batch)
    # Perform parameter update with gradients and optimizer
    state = state.apply_gradients(grads=grads)
    # Return state and any other value we might want
    metrics = {'train_loss': loss,
               'train_target': jnp.argmax(batch[1]),
               "train_output": jnp.mean(output),
               "train_var_output": jnp.std(output)}
    return state, rng, metrics


@jax.jit
def test_step(state: TrainState, batch, baseline):
    xz_data, z_target = batch
    output = state.apply_fn({'params': state.params}, xz_data, train=False)
    ce = ce_loss(output, z_target)
    #rcd_ce = _rcd_pred(xz_data)
    metrics = {"test_loss": ce,
               "test_target": jnp.argmax(batch[1]),
               "test_output": jnp.mean(output),
               "test_val_output": jnp.std(output),
               "baseline_loss": ce_loss(baseline, z_target),
               }
    return metrics


def init(model: TransitivePredictor, seed: int, train_dataloader: Iterable, lr: float = 1e-4, grad_clip_value: float = 1,
         accumulate_grads: int = 1):
    rng = jax.random.PRNGKey(seed)
    xz_data, z_target = next(iter(train_dataloader))
    rng, init_rng, dropout_init_rng = jax.random.split(rng, 3)
    params = model.init({'params': init_rng, 'dropout': dropout_init_rng}, xz_data)['params']
    optimiser = get_optimiser(lr, grad_clip_value, accumulate_grads)
    opt_state = optimiser.init(params)
    model_state = TrainState(step=0, apply_fn=model.apply, params=params, tx=optimiser, opt_state=opt_state)

    return model_state, rng


def train(model: TransitivePredictor,
          train_dataloader: Iterable,
          val_dataloader: Iterable = None,
          max_epochs: int = 1,
          log_every_n_step: int = 25,
          val_every_n_steps: int = 1000,
          ckpt_every_n_steps: int = 10000,
          log_dir: str = None,
          lr: float = 1e-4,
          grad_clip_value: float = 1,
          accumulate_grads: int = 1,
          seed: int = 42
          ) -> Tuple[TrainState, Callable]:

    if log_dir is None:
        log_dir = get_default_log_dir()
        logger = SummaryWriter(log_dir)

    model_state, rng = init(model, seed, train_dataloader, lr, grad_clip_value, accumulate_grads)
    grad_step = 0
    val_epoch = 0
    val_step = 0

    print('Calc Baseline')
    targets = jnp.asarray([t for e, t in train_dataloader])
    baseline = jnp.mean(targets, axis=0)
    jnp.save(os.path.join(log_dir, 'baseline_prediction.npy'), baseline)
    #try:
    for j in range(max_epochs):
        for i, batch in enumerate(tqdm(train_dataloader, desc="Train Epoch {}".format(j))):
            model_state, rng, metrics = train_step(model_state, rng, batch)
            grad_step += 1
            if i % log_every_n_step == 0:
                _log_metrics(logger, metrics, grad_step)
            if i % ckpt_every_n_steps == 0:
                checkpoints.save_checkpoint(log_dir, model_state, grad_step)
            if i % val_every_n_steps == 0 and val_dataloader is not None:
                val_epoch += 1
                for val_batch in tqdm(val_dataloader, desc="Val Step {}".format(i)):
                    metrics = test_step(model_state, val_batch, baseline)
                    val_step += 1
                    _log_metrics(logger, metrics, val_step)
    #except Exception as e:
    #    print(e)
    #    print('Exception occurred during training. Saving model state.')
    #    checkpoints.save_checkpoint(log_dir, model_state, grad_step)
    checkpoints.save_checkpoint(log_dir, model_state, grad_step)
    return model_state


def _log_metrics(logger, metrics: Dict[str, Any], step: int):
    for name, val in metrics.items():
        logger.scalar(name, val, step)
        logger.flush()


def _rcd_pred(XZ_data):
    eps = 1e-5
    res = infer_bivariate_structure(XZ_data)
    if res == 'indep':
        return jnp.asarray([1-2*eps, 0, 0, 0]) + eps
    elif res == '<->':
        return jnp.asarray([0, 1-2*eps, 0, 0]) + eps
    elif res == '->':
        return jnp.asarray([0, 0, 1-2*eps, 0]) + eps
    elif res == '<-':
        return jnp.asarray([0, 0, 0, 1-2*eps]) + eps
    else:
        raise NotImplementedError()