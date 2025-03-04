import flax.linen as nn
import jax
import jax.numpy as jnp


class FeedForward(nn.Module):

    hidden_dim: int
    widening_factor: int = 4
    depth: int = 1
    dropout_rate: float = 0.
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, input_array: jax.Array, train: bool = False) -> jax.Array:
        h = input_array
        for _ in range(self.depth):
            h = nn.Dense(features=self.widening_factor * self.hidden_dim, dtype=self.dtype)(h)
            h = nn.gelu(h)
            h = nn.Dropout(self.dropout_rate)(h, deterministic=not train)
        h = nn.Dense(features=self.hidden_dim, dtype=self.dtype)(h)
        h = nn.gelu(h)
        h = nn.Dropout(self.dropout_rate)(h, deterministic=not train)
        return h
