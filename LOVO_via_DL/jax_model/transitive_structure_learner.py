from typing import Optional

import jax
import jax.numpy as jnp
import flax.linen as nn
from jax_model.feed_forward import FeedForward


class TransitiveStructureLearner(nn.Module):
    """Learn to predict causal structure from embeddings of TransitivePredictor

       Parameters
    ----------
    num_heads: int
        Number of heads in multi-head attention in the alternate attention blocks.
    num_layers: int
        Number of stacked alternate attention blocks.
    hidden_dim: int
        Hidden dimension of the linear layers in multi-head attention. This corresponds to the
        embedding dimension.
    dropout_rate: float
        Dropout probability in multi-head attention in the alternate attention blocks.
    widening_factor: int
        Factor by which the MLP hidden layer widens.
    rff_depth: int
        Depth of the row-wise feed forward network after the attention block
    dtype: jnp.dtype
        Data type used in the model
    eps_layer_norm : float
        Small value added at each application of layer norm for numerical stability
    name: str
        Optional name to indicate layer
    """
    #TODO most params unused
    max_num_nodes: int
    num_heads: int
    depth_encoder: int
    depth_decoder: int
    hidden_dim: int
    dropout_rate: float
    lr: float
    grad_clip_value: float = 1
    widening_factor: int = 4
    rff_depth: int = 1
    dtype: jnp.dtype = jnp.float32
    eps_layer_norm: float = 1e-05
    name: Optional[str] = None
    start_token: float = -1
    threshold: float = .5
    accumulate_grads: int = 1

    def setup(self):
        self.predictor = FeedForward(hidden_dim=self.hidden_dim, widening_factor=4,
                                     depth=self.depth_decoder, dtype=self.dtype, name="Predictor")
        self.out = nn.Dense(features=4)

    def __call__(self, xz_data, train: bool = False) -> jax.Array:
        output = self.out(self.predictor(xz_data, train=train).flatten())
        return output.flatten()

