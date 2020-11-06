# implemented as functions as keras/tf had problems loading models with custom layers in custom layers
# (wrong weight order)
import tensorflow as tf
import numpy as np
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Dense, Lambda, LayerNormalization, Embedding, Layer, Dropout, Reshape, add, Activation, Conv1D
from tensorflow.keras.activations import softmax
from tensorflow.keras.utils import get_custom_objects


def add_mhs_attention(inputs, embed_dim, num_heads, initializer, current_tformer):
    if embed_dim % num_heads != 0:
        raise ValueError(
            f"embedding dimension = {embed_dim} should be divisible by number of heads = {num_heads}"
        )
    projection_dim = embed_dim // num_heads
    query_dense = Dense(embed_dim, kernel_initializer=initializer)
    key_dense = Dense(embed_dim, kernel_initializer=initializer)
    value_dense = Dense(embed_dim, kernel_initializer=initializer)
    combine_heads = Dense(embed_dim, kernel_initializer=initializer)
    seq_len = inputs.shape[1]

    def separate_heads(input, qkv_name):
        out = Reshape((seq_len, num_heads, projection_dim))(input)
        perm = Lambda(lambda x: K.permute_dimensions(x, pattern=[0, 2, 1, 3]),
                      name="permute_attention_{qkv}_{n}".format(qkv=qkv_name, n=current_tformer))
        out = perm(out)
        return out

    def get_attention(query, key, value):
        score = tf.matmul(query, key, transpose_b=True)
        dim_key = key.shape[-1]
        scale = Lambda(lambda x: x / np.sqrt(dim_key),
                       name="scale_attention_{n}".format(n=current_tformer))
        scaled_score = scale(score)
        weights = Activation(softmax)(scaled_score)
        output = tf.matmul(weights, value)
        return output, weights

    query = query_dense(inputs)
    key = key_dense(inputs)
    value = value_dense(inputs)
    query = separate_heads(query, "query")
    key = separate_heads(key, "key")
    value = separate_heads(value, "value")
    att_out, weights = get_attention(query, key, value)
    perm = Lambda(lambda x: K.permute_dimensions(x, pattern=[0, 2, 1, 3]),
                  name="permute_attention_out_{n}".format(n=current_tformer))
    att_out = perm(att_out)
    att_out = Reshape((seq_len, embed_dim))(att_out)
    output = combine_heads(att_out)
    return output


def add_transformer_block(inputs, embed_dim, num_heads, ff_dim, dropout_rate, initializer, current_tformer,
                          training=False):
    attn_output = add_mhs_attention(inputs, embed_dim, num_heads, initializer, current_tformer)

    layernorm1 = LayerNormalization(epsilon=1e-6)
    layernorm2 = LayerNormalization(epsilon=1e-6)
    dropout1 = Dropout(dropout_rate)
    dropout2 = Dropout(dropout_rate)

    if not np.isclose(dropout_rate, 0.0):
        attn_output = dropout1(attn_output, training=training)
    out1 = layernorm1(add([inputs, attn_output]))
    ffn_output = Dense(ff_dim, activation="relu", kernel_initializer=initializer)(out1)
    ffn_output = Dense(embed_dim, kernel_initializer=initializer)(ffn_output)
    if not np.isclose(dropout_rate, 0.0):
        ffn_output = dropout2(ffn_output, training=training)
    return layernorm2(add([out1, ffn_output]))


class PositionEmbedding(Layer):
    def __init__(self, max_depth, seed, use_depth=True, **kwargs):
        self.max_depth = max_depth
        self.horizontal_position_embeddings = None
        self.vertical_position_embeddings = None
        self.seed = seed
        self.use_depth = False if max_depth == 1 else use_depth
        self.initializer = tf.keras.initializers.RandomUniform(seed=self.seed)
        super().__init__(**kwargs)

    def get_config(self):
        config = super().get_config()
        config['max_depth'] = self.max_depth
        config['seed'] = self.seed
        config['use_depth'] = self.use_depth
        return config

    def build(self, input_shape):
        seq_length, embed_dim = input_shape[-2:]
        self.horizontal_position_embeddings = self.add_weight(
            shape=(seq_length, embed_dim),
            initializer=self.initializer,
            name='horizontal_position_embeddings',
            trainable=True)
        if self.use_depth:
            self.vertical_position_embeddings = self.add_weight(
                shape=(self.max_depth, embed_dim),
                initializer=self.initializer,
                name='vertical_position_embeddings',
                trainable=True)
        super().build(input_shape)

    def call(self, inputs, **kwargs):
        current_tformer = kwargs.get('current_tformer')
        out = inputs + self.horizontal_position_embeddings
        if self.use_depth:
            out = out + self.vertical_position_embeddings[current_tformer, :]
        return out


get_custom_objects().update({
    'PositionEmbedding': PositionEmbedding,
})
