import jax
import jax.numpy as jnp
from flax.linen import MultiHeadDotProductAttention
import flax.linen as nn

from jax_model.alternate_attention import AlternatingAttentionStack


class CausalEncoder(nn.Module):
    """
    Encoder part of the model as described in 'Learning to induce causal structure' by Ke et al.

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
        self.input_embedding = [
            nn.Dense(features=int(self.hidden_dim / 2), dtype=self.dtype, name="InputEnc{}".format(i))
            for i in range(self.max_num_nodes + 1)
        ]
        self.identity_embedding = [
            nn.Dense(features=int(self.hidden_dim / 2), dtype=self.dtype, name="IdEnc{}".format(i))
            for i in range(self.max_num_nodes + 1)
        ]
        self.attention_stack = AlternatingAttentionStack(max_num_nodes=self.max_num_nodes,
                                                         num_heads=self.num_heads,
                                                         num_layers=self.num_layers,
                                                         hidden_dim=self.hidden_dim,
                                                         dropout_rate=self.dropout_rate,
                                                         widening_factor=self.widening_factor,
                                                         rff_depth=self.rff_depth,
                                                         dtype=self.dtype,
                                                         eps_layer_norm=self.eps_layer_norm,
                                                         name="EncAttnStack"
                                                         )
        self.summary_attention = MultiHeadDotProductAttention(num_heads=1,
                                                              dtype=self.dtype,
                                                              dropout_rate=self.dropout_rate,
                                                              name="SummaryAttn"
                                                              )

    def __call__(self,
                 data: jax.Array,  # input data, shape [N, D].
                 train: bool = True
                 ) -> jax.Array:  # Batch of sequences of output token logits, shape [N, D].
        """Forward pass, producing a sequence of logits."""

        # Embed every dataset[node, sample] with node-specific embedding. Output shape N x D X E / 2
        num_nodes = data.shape[1]
        input_embedding = jnp.stack([
            embed(jnp.expand_dims(data[:, i], -1)) for i, embed in enumerate(self.input_embedding[:num_nodes])
        ], axis=1
        )  # [N, D, E / 2]
        # Compute the index embeddings and sum them to X. Output shape D x E / 2
        features_id = jnp.arange(num_nodes)
        identity_embedding = jnp.expand_dims(jnp.stack([
            embed(jnp.expand_dims(features_id[i], -1)) for i, embed in enumerate(self.identity_embedding[:num_nodes])
        ], axis=0
        ), axis=0)  # [1, D, E / 2]
        # Sum index input embeddings
        embedding = jnp.concatenate([input_embedding, jnp.repeat(identity_embedding, repeats=input_embedding.shape[0], axis=0)], axis=-1)    # [N, D, E]
        # Add summary entry
        embedding, datapoint_mask = self._add_summary_dim(embedding)    # [N+1, D, E], [D, N+1]

        # Run the transformer over the inputs. The alternating attention expects input with [N+1, D, E]
        embedding = self.attention_stack(embedding, datapoint_mask=datapoint_mask, train=train)
        embedding = jnp.transpose(embedding, (1, 0, 2))  # [D, N+1, E]
        # Summarize along data axis
        summary = self._encoder_summary(embedding, train=train)  # [D, E] TODO check if faster if dims are transposed
        return summary

    def _encoder_summary(
            self,
            embedding: jax.Array,
            train: bool = True
    ):
        """Output the encoder summary as weighted average of the X samples.

        Parameters
        ----------
        embedding : Tensor of shape (num_nodes, num_samples + 1, embed_dim)
            Tensor transformed by multiple stacks of alternate attention

        Returns
        -------
        Tensor of shape (num_nodes, embed_dim).
        Weighted average of X entries across the 'num_samples' dimension.
        The weights are defined as a vector of shape (num_samples, ), formed
        by key-value attention where the query the summary entry X[:, -1]
        and the key values are the transformed samples X[:, :-1]
        """
        query = jnp.expand_dims(embedding[:, -1, :], 1)
        keys = embedding[:, :-1, :]
        values = embedding[:, :-1, :]

        values_avg = self.summary_attention(query, keys, values, deterministic=not train)
        return values_avg[:, 0, :]

    def _add_summary_dim(
            self,
            embedding: jax.Array
    ):
        """Add column to deposit summary info to the second dimension of the input tensor.

        Parameters
        ----------
        embedding : Tensor
            Dataset tensor of shape (num_nodes, num_samples, embed_dim).
           `embedding` is modified by adding a zero vector of shape `num_nodes` to axis=1.

        Returns
        -------
        embedding: Tensor
        datapoints_mask: Tensor
            Boolean or float mask tensor to prevent attending to the added summary entries.
        """
        num_samples, num_nodes, _ = embedding.shape

        # Create mask (attn weights are multiplied by the mask values)
        datapoints_mask = jnp.zeros((num_nodes, num_samples+1), dtype=embedding.dtype)  # exp^0 = 1
        datapoints_mask = datapoints_mask.at[:, -1].set(1)
        # datapoints mask is applied to attention across datapoints, i.e. num nodes is the batch dim.
        # jax expects mask to be batch x num_heads x num_samples+1 x num_samples+1
        datapoints_mask = jnp.matmul(jnp.expand_dims(datapoints_mask, -1), jnp.expand_dims(datapoints_mask, -2))
        datapoints_mask = jnp.expand_dims(datapoints_mask, 1)

        # jax expects 0 for masked entries
        datapoints_mask = 1 - datapoints_mask
        #TODO DEBUG
        datapoints_mask = jnp.zeros(num_samples + 1)
        datapoints_mask = datapoints_mask.at[-1].set(1)
        datapoints_mask = jnp.outer(datapoints_mask, datapoints_mask)
        datapoints_mask = 1 - datapoints_mask

        # Add dimension
        # TODO: test if summary entry != 0 changes the result.
        summary_entry = jnp.zeros((1, num_nodes, self.hidden_dim))
        embedding = jnp.concatenate([embedding, summary_entry], axis=0)
        return embedding, datapoints_mask
