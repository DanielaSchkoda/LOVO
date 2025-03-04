import os

import pandas as pd
from flax.training import checkpoints
from tqdm import tqdm
import jax.numpy as jnp
from jax_model.transitive_structure_learner import TransitiveStructureLearner
from training_functions_structure import test_step, ce_loss
from graph_dataset import StructureDatasetGenerator

# Training hyperparameters.
LEARNING_RATE = 1e-4
GRAD_CLIP_VALUE = 10
ACC_GRADS = 1
LOG_EVERY = 1
VAL_EVERY = 50
CKPT_EVERY = 1024
N_EPOCHS = 10
SEED = 0

# Model hyperparameters.
NUM_ENC_LAYERS = 3
NUM_DEC_LAYERS = 1
NUM_HEADS = 8  # Number of attention heads.
MODEL_SIZE = 64
DROPOUT_RATE = 0.0

# Data parameters
NUM_DATASETS_TEST = 100
NUM_SAMPLES = 3000

if __name__ == '__main__':
    test_dataset = StructureDatasetGenerator(num_datasets=NUM_DATASETS_TEST, num_samples=NUM_SAMPLES)
    model = TransitiveStructureLearner(
        max_num_nodes=2,
        hidden_dim=MODEL_SIZE,
        depth_encoder=NUM_ENC_LAYERS,
        depth_decoder=NUM_DEC_LAYERS,
        dropout_rate=DROPOUT_RATE, num_heads=NUM_HEADS,
        lr=LEARNING_RATE,
        accumulate_grads=ACC_GRADS
    )
    model_state = checkpoints.restore_checkpoint('checkpoint_structure', target=None)

    baseline = jnp.load(os.path.join('checkpoint_structure', 'baseline_prediction.npy'))
    metrics = []
    for embedding, label in tqdm(test_dataset, desc="Val Step"):
        pred = model.apply({'params': model_state['params']}, embedding)
        metrics.append({'DL': ce_loss(pred, label).item(), 'Baseline': ce_loss(baseline, label).item()})

    df = pd.DataFrame(metrics)
    df.to_csv('structure_test_res.csv')
    print(df.mean())
    print(df.std())