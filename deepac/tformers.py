# based on https://keras.io/examples/nlp/text_classification_with_transformer/ by Apoorv Nandan
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers


class MultiHeadSelfAttention(layers.Layer):
    def __init__(self, embed_dim, num_heads, initializer, **kwargs):
        super(MultiHeadSelfAttention, self).__init__(kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.initializer = initializer
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embedding dimension = {embed_dim} should be divisible by number of heads = {num_heads}"
            )
        self.projection_dim = embed_dim // num_heads
        self.query_dense = layers.Dense(embed_dim, kernel_initializer=self.initializer)
        self.key_dense = layers.Dense(embed_dim, kernel_initializer=self.initializer)
        self.value_dense = layers.Dense(embed_dim, kernel_initializer=self.initializer)
        self.combine_heads = layers.Dense(embed_dim, kernel_initializer=self.initializer)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'embed_dim': self.embed_dim,
            'num_heads': self.num_heads,
            "initializer": keras.initializers.serialize(self.initializer)
        })
        return config

    def attention(self, query, key, value):
        score = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_score = score / tf.math.sqrt(dim_key)
        weights = tf.nn.softmax(scaled_score, axis=-1)
        output = tf.matmul(weights, value)
        return output, weights

    def separate_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs, training=False):
        # x.shape = [batch_size, seq_len, embedding_dim]
        batch_size = tf.shape(inputs)[0]
        query = self.query_dense(inputs)  # (batch_size, seq_len, embed_dim)
        key = self.key_dense(inputs)  # (batch_size, seq_len, embed_dim)
        value = self.value_dense(inputs)  # (batch_size, seq_len, embed_dim)
        query = self.separate_heads(
            query, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)
        key = self.separate_heads(
            key, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)
        value = self.separate_heads(
            value, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)
        attention, weights = self.attention(query, key, value)
        attention = tf.transpose(
            attention, perm=[0, 2, 1, 3]
        )  # (batch_size, seq_len, num_heads, projection_dim)
        concat_attention = tf.reshape(
            attention, (batch_size, -1, self.embed_dim)
        )  # (batch_size, seq_len, embed_dim)
        output = self.combine_heads(
            concat_attention
        )  # (batch_size, seq_len, embed_dim)
        return output


class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout_rate, initializer, **kwargs):
        super(TransformerBlock, self).__init__(kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate
        self.initializer = initializer
        self.att = MultiHeadSelfAttention(embed_dim, num_heads, initializer)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu", kernel_initializer=self.initializer),
             layers.Dense(embed_dim, kernel_initializer=self.initializer), ]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'dropout_rate': self.dropout_rate,
            'ff_dim': self.ff_dim,
            'embed_dim': self.embed_dim,
            'num_heads': self.num_heads,
            "initializer": keras.initializers.serialize(self.initializer)
        })
        return config

    def call(self, inputs, training=False):
        attn_output = self.att(inputs)
        if not np.isclose(self.dropout_rate, 0.0):
            attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        if not np.isclose(self.dropout_rate, 0.0):
            ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim, token_identity=True, **kwargs):
        super(TokenAndPositionEmbedding, self).__init__(kwargs)
        self.maxlen = maxlen
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.token_identity = token_identity
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim) if not token_identity else None
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'vocab_size': self.vocab_size,
            'maxlen': self.maxlen,
            'embed_dim': self.embed_dim,
            'token_identity': self.token_identity
        })
        return config

    def call(self, inputs, training=False):
        positions = tf.range(start=0, limit=self.maxlen, delta=1)
        positions = self.pos_emb(positions)
        if not self.token_identity:
            inputs = self.token_emb(inputs)
        return inputs + positions

