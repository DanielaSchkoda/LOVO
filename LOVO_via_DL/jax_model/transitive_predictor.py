from typing import Optional

import jax
import jax.numpy as jnp
import flax.linen as nn
from jax_model.encoder import CausalEncoder
from jax_model.feed_forward import FeedForward


class TransitivePredictor(nn.Module):
    """CSIVA architecture.
        A single datapoint for the supervised training of the model is an entire dataset,
        of shape (num_samples, num_nodes).

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
        self.encoder = CausalEncoder(max_num_nodes=self.max_num_nodes,
                                     num_heads=self.num_heads,
                                     num_layers=self.depth_encoder,
                                     hidden_dim=self.hidden_dim,
                                     dropout_rate=self.dropout_rate,
                                     widening_factor=self.widening_factor,
                                     rff_depth=self.rff_depth,
                                     dtype=self.dtype,
                                     eps_layer_norm=self.eps_layer_norm,
                                     name="Enc"
                                     )
        self.predictor = FeedForward(hidden_dim=self.hidden_dim, widening_factor=4,
                                     depth=self.depth_decoder, dtype=self.dtype, name="Predictor")
        self.out = nn.Dense(features=1)

    def __call__(self, xy_data, xz_data, train: bool = False) -> jax.Array:
        xy_mem = self.encoder(xy_data, train=train)
        xz_mem = self.encoder(xz_data, train=train)
        pred_input = jnp.concatenate((xy_mem.flatten(), xz_mem.flatten()))
        output = self.out(self.predictor(pred_input, train=train))
        return output.flatten()

    def embedd(self, xz_data, train: bool = False) -> jax.Array:
        return self.encoder(xz_data, train=train)

