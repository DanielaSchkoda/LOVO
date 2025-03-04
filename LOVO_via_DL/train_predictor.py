import time

from jax_model.transitive_predictor import TransitivePredictor
from training_functions import train
from graph_dataset import GraphSplitDatasetGenerator

# Training hyperparameters.
LEARNING_RATE = 1e-4
GRAD_CLIP_VALUE = 100
ACC_GRADS = 1
LOG_EVERY = 1
VAL_EVERY = 50
CKPT_EVERY = 1024
N_EPOCHS = 50
SEED = 0

# Model hyperparameters.
NUM_ENC_LAYERS = 3
NUM_DEC_LAYERS = 1
NUM_HEADS = 8  # Number of attention heads.
MODEL_SIZE = 64
DROPOUT_RATE = 0.0

# Data parameters
NUM_DATASETS = 1000
NUM_DATASETS_TEST = 50
NUM_SAMPLES = 300

if __name__ == '__main__':
    # Create the dataset.
    train_dataset = GraphSplitDatasetGenerator(num_datasets=NUM_DATASETS, num_samples=NUM_SAMPLES)
    val_dataset = GraphSplitDatasetGenerator(num_datasets=NUM_DATASETS_TEST, num_samples=NUM_SAMPLES)

    start_time = time.time()
    model = TransitivePredictor(
        max_num_nodes=2,
        hidden_dim=MODEL_SIZE,
        depth_encoder=NUM_ENC_LAYERS,
        depth_decoder=NUM_DEC_LAYERS,
        dropout_rate=DROPOUT_RATE, num_heads=NUM_HEADS,
        lr=LEARNING_RATE,
        accumulate_grads=ACC_GRADS
    )

    train(model=model,
          train_dataloader=train_dataset,
          val_dataloader=val_dataset,
          max_epochs=N_EPOCHS,
          log_every_n_step=LOG_EVERY,
          val_every_n_steps=VAL_EVERY,
          ckpt_every_n_steps=CKPT_EVERY,
          lr=LEARNING_RATE,
          grad_clip_value=GRAD_CLIP_VALUE,
          accumulate_grads=ACC_GRADS,
          seed=SEED)

    end_time = time.time()
    elapsed_time_seconds = end_time - start_time
    # Convert seconds to hours, minutes, and seconds
    hours, remainder = divmod(elapsed_time_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    # Print the elapsed time nicely
    print(f"Elapsed time: {int(hours)} hours, {int(minutes)} minutes, {seconds:.2f} seconds")
