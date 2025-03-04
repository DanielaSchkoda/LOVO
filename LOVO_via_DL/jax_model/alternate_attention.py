import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.linen import MultiHeadDotProductAttention

from jax_model.feed_forward import FeedForward


class AlternatingAttentionBlock(nn.Module):
    """Multi Head Attention Block as used by Kossen et al. in their "Non-parametric Transformer"

        Parameters
    ----------
    num_heads: int
        Number of heads in multi-head attention in the alternate attention blocks.
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

    num_heads: int
    hidden_dim: int
    dropout_rate: float
    widening_factor: int = 4
    rff_depth: int = 1
    dtype: jnp.dtype = jnp.float32
    eps_layer_norm: float = 1e-05

    def setup(self):
        self.pre_att_norm = nn.LayerNorm(dtype=self.dtype, epsilon=self.eps_layer_norm)
        self.pre_dense_norm = nn.LayerNorm(dtype=self.dtype, epsilon=self.eps_layer_norm)
        self.layer_norm = nn.LayerNorm(dtype=self.dtype, epsilon=self.eps_layer_norm)
        self.attn = MultiHeadDotProductAttention(num_heads=self.num_heads,
                                                 dtype=self.dtype,
                                                 dropout_rate=self.dropout_rate,
                                                 name="AttnBlock"
                                                 )
        self.rff = FeedForward(hidden_dim=self.hidden_dim,
                               widening_factor=self.widening_factor,
                               depth=self.rff_depth,
                               dtype=self.dtype, name="rFF"
                               )
        self.w_res = nn.Dense(features=self.hidden_dim, dtype=self.dtype)

    def __call__(self,
                 embeddings: jax.Array,
                 train: bool = True,
                 mask: jax.Array = None,
                 ) -> jax.Array:  # [D, N+1, E]

        h = embeddings
        # Eq. 4 in Kossen et al.
        h_norm = self.pre_att_norm(h)
        h_attn = self.attn(h_norm, h_norm, h_norm, deterministic=not train, mask=mask)
        #print(h_attn.shape, self.w_res(h).shape)
        h_res = self.w_res(h) + h_attn
        # Eq. 5
        h_dense = self.rff(self.pre_dense_norm(h_res))
        return h_res + h_dense


class AlternatingAttentionLayer(nn.Module):
    """Alternating attention between datapoints and attributes.

        Parameters
    ----------
    num_heads: int
        Number of heads in multi-head attention in the alternate attention blocks.
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
    hidden_dim: int
    dropout_rate: float
    widening_factor: int = 4
    rff_depth: int = 1
    dtype: jnp.dtype = jnp.float32
    eps_layer_norm: float = 1e-05

    def setup(self):
        self.att_between_attributes = AlternatingAttentionBlock(num_heads=self.num_heads,
                                                                hidden_dim=self.hidden_dim,
                                                                dropout_rate=self.dropout_rate,
                                                                widening_factor=self.widening_factor,
                                                                rff_depth=self.rff_depth,
                                                                dtype=self.dtype,
                                                                eps_layer_norm=self.eps_layer_norm,
                                                                name="ABAttr"
                                                                )
        self.att_between_datapoints = AlternatingAttentionBlock(num_heads=self.num_heads,
                                                                hidden_dim=self.hidden_dim * self.max_num_nodes,
                                                                dropout_rate=self.dropout_rate,
                                                                widening_factor=self.widening_factor,
                                                                rff_depth=self.rff_depth,
                                                                dtype=self.dtype,
                                                                eps_layer_norm=self.eps_layer_norm,
                                                                name="ABData"
                                                                )

    def __call__(self,
                 embeddings: jax.Array,  # [N+1, D, E]
                 datapoint_mask: jax.Array = None,  # [D, N+1]
                 train: bool = True
                 ) -> jax.Array:  # [N+1, D, E]

        h = embeddings
        # expect dim N+1 x D x E
        n, _, e = h.shape
        h = jnp.reshape(h, (n, self.max_num_nodes * e))
        h = self.att_between_datapoints(h, train=train, mask=datapoint_mask)

        h = jnp.reshape(h, (n, self.max_num_nodes, e))  # N+1 x D x E
        h = self.att_between_attributes(h, train=train)
        return h


class AlternatingAttentionStack(nn.Module):
    """
    Stacked layer of alternating attention blocks.

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
    num_layers: int
    hidden_dim: int
    dropout_rate: float
    widening_factor: int = 4
    rff_depth: int = 1
    dtype: jnp.dtype = jnp.float32
    eps_layer_norm: float = 1e-05

    def setup(self):
        self.layers = [AlternatingAttentionLayer(max_num_nodes=self.max_num_nodes,
                                                 num_heads=self.num_heads,
                                                 hidden_dim=self.hidden_dim,
                                                 dropout_rate=self.dropout_rate,
                                                 widening_factor=self.widening_factor,
                                                 rff_depth=self.rff_depth,
                                                 dtype=self.dtype,
                                                 eps_layer_norm=self.eps_layer_norm,
                                                 name="AltAtt_{}".format(i)
                                                 ) for i in range(self.num_layers)
                       ]

    def __call__(
            self,
            embeddings: jax.Array,  # [N, D, E]
            datapoint_mask: jax.Array = None,  # [N, D]
            train: bool = True
    ) -> jax.Array:  # [B, T, D]
        """Transforms input embedding sequences to output embedding sequences."""

        # TODO add back. But be careful which mask to use for which attention
        #  if key_padding_mask is not None:
        #    mask = key_padding_mask[:, None, None, :]  # [B, H=1, D'=1, D]
        #    # set causal mask to all ones, as we want to encode all data
        #    causal_mask = np.ones((1, 1, num_nodes, num_nodes))  # [B=1, H=1, D, D]
        #    mask = mask * causal_mask  # [B, H=1, D, D]
        for layer in self.layers:
            embeddings = layer(embeddings, datapoint_mask=datapoint_mask, train=train)
        return embeddings
