# partially inspired by
# https://keras.io/examples/nlp/text_classification_with_transformer/ (by Apoorv Nandan)
# and https://github.com/kpot/keras-transformer/blob/master/keras_transformer/position.py (by Kirill Mavreshko)
# implemented as functions since keras/tf had problems loading models with custom layers in custom layers
# (wrong weight order)
import tensorflow as tf
import numpy as np
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Dense, Lambda, LayerNormalization, Layer, Dropout,\
    Reshape, add, Activation, concatenate
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


def add_siam_mhs_attention(inputs_fwd, inputs_rc, embed_dim, num_heads, initializer, current_tformer):
    if embed_dim % num_heads != 0:
        raise ValueError(
            f"embedding dimension = {embed_dim} should be divisible by number of heads = {num_heads}"
        )
    projection_dim = embed_dim // num_heads
    query_dense = Dense(embed_dim, kernel_initializer=initializer)
    key_dense = Dense(embed_dim, kernel_initializer=initializer)
    value_dense = Dense(embed_dim, kernel_initializer=initializer)
    combine_heads = Dense(embed_dim, kernel_initializer=initializer)
    seq_len = inputs_fwd.shape[1]

    def separate_heads(input, qkv_name):
        out = Reshape((seq_len, num_heads, projection_dim))(input)
        perm = Lambda(lambda x: K.permute_dimensions(x, pattern=[0, 2, 1, 3]),
                      name="permute_attention_{qkv}_{n}".format(qkv=qkv_name, n=current_tformer))
        out = perm(out)
        return out

    def get_attention(query, key, value, mode_name=""):
        score = tf.matmul(query, key, transpose_b=True)
        dim_key = key.shape[-1]
        scale = Lambda(lambda x: x / np.sqrt(dim_key),
                       name="scale_attention_{mode}_{n}".format(mode=mode_name, n=current_tformer))
        scaled_score = scale(score)
        weights = Activation(softmax)(scaled_score)
        output = tf.matmul(weights, value)
        return output, weights

    query_fwd = query_dense(inputs_fwd)
    key_fwd = key_dense(inputs_fwd)
    value_fwd = value_dense(inputs_fwd)
    query_rc = query_dense(inputs_rc)
    key_rc = key_dense(inputs_rc)
    value_rc = value_dense(inputs_rc)
    query_fwd = separate_heads(query_fwd, "query_fwd")
    key_fwd = separate_heads(key_fwd, "key_fwd")
    value_fwd = separate_heads(value_fwd, "value_fwd")
    query_rc = separate_heads(query_rc, "query_rc")
    key_rc = separate_heads(key_rc, "key_rc")
    value_rc = separate_heads(value_rc, "value_rc")
    att_out_fwd, weights_fwd = get_attention(query_fwd, key_fwd, value_fwd, "fwd")
    att_out_rc, weights_rc = get_attention(query_rc, key_rc, value_rc, "rc")
    perm_fwd = Lambda(lambda x: K.permute_dimensions(x, pattern=[0, 2, 1, 3]),
                      name="permute_attention_fwd_out_{n}".format(n=current_tformer))
    att_out_fwd = perm_fwd(att_out_fwd)
    att_out_fwd = Reshape((seq_len, embed_dim))(att_out_fwd)
    output_fwd = combine_heads(att_out_fwd)
    perm_rc = Lambda(lambda x: K.permute_dimensions(x, pattern=[0, 2, 1, 3]),
                     name="permute_attention_rc_out_{n}".format(n=current_tformer))
    att_out_rc = perm_rc(att_out_rc)
    att_out_rc = Reshape((seq_len, embed_dim))(att_out_rc)
    output_rc = combine_heads(att_out_rc)
    return output_fwd, output_rc


def add_transformer_block(inputs, embed_dim, position_embedding, num_heads, ff_dim, dropout_rate, initializer,
                          current_tformer, seed, training=False):

    inputs = position_embedding(inputs, current_tformer=current_tformer)
    attn_output = add_mhs_attention(inputs, embed_dim, num_heads, initializer, current_tformer)

    layernorm1 = LayerNormalization(epsilon=1e-6)
    layernorm2 = LayerNormalization(epsilon=1e-6)

    if not np.isclose(dropout_rate, 0.0):
        attn_output = Dropout(dropout_rate, seed=seed)(attn_output, training=training)
    out1 = layernorm1(add([inputs, attn_output]))
    ffn_output = Dense(ff_dim, activation="relu", kernel_initializer=initializer)(out1)
    ffn_output = Dense(embed_dim, kernel_initializer=initializer)(ffn_output)
    if not np.isclose(dropout_rate, 0.0):
        ffn_output = Dropout(dropout_rate, seed=seed)(ffn_output, training=training)
    return layernorm2(add([out1, ffn_output]))


def add_siam_transformer_block(inputs_fwd, inputs_rc, position_embedding, embed_dim, num_heads, ff_dim, dropout_rate,
                               initializer, current_tformer, seed, training=False):

    inputs_fwd = position_embedding(inputs_fwd, current_tformer=current_tformer)
    inputs_rc = position_embedding(inputs_rc, current_tformer=current_tformer)
    attn_output_fwd, attn_output_rc = add_siam_mhs_attention(inputs_fwd, inputs_rc, embed_dim, num_heads, initializer,
                                                             current_tformer)

    if not np.isclose(dropout_rate, 0.0):
        attn_output_fwd = Dropout(dropout_rate, seed=seed)(attn_output_fwd, training=training)
        attn_output_rc = Dropout(dropout_rate, seed=seed)(attn_output_rc, training=training)
    out1_fwd, out1_rc = add_siam_layernorm(add([inputs_fwd, attn_output_fwd]), add([inputs_rc, attn_output_rc]),
                                           current_ln=current_tformer*2)
    ffn_1 = Dense(ff_dim, activation="relu", kernel_initializer=initializer)
    ffn_2 = Dense(embed_dim, kernel_initializer=initializer)
    ffn_output_fwd = ffn_1(out1_fwd)
    ffn_output_fwd = ffn_2(ffn_output_fwd)
    ffn_output_rc = ffn_1(out1_rc)
    ffn_output_rc = ffn_2(ffn_output_rc)
    if not np.isclose(dropout_rate, 0.0):
        ffn_output_fwd = Dropout(dropout_rate, seed=seed)(ffn_output_fwd, training=training)
        ffn_output_rc = Dropout(dropout_rate, seed=seed)(ffn_output_rc, training=training)
    output_fwd, output_rc = add_siam_layernorm(add([out1_fwd, ffn_output_fwd]), add([out1_rc, ffn_output_rc]),
                                               current_ln=(current_tformer*2)+1)
    return output_fwd, output_rc


def add_rc_transformer_block(inputs, embed_dim, position_embedding, num_heads, ff_dim, dropout_rate, initializer,
                             current_tformer, seed, training=False):

    revcomp_in = Lambda(lambda x: K.reverse(x, axes=(1, 2)), output_shape=inputs.shape[1:],
                        name="reverse_complement_tformer_input_{n}".format(n=current_tformer))
    inputs_rc = revcomp_in(inputs)
    output_fwd, output_rc = add_siam_transformer_block(inputs, inputs_rc, position_embedding, embed_dim, num_heads,
                                                       ff_dim, dropout_rate, initializer, current_tformer, seed,
                                                       training)

    revcomp_out = Lambda(lambda x: K.reverse(x, axes=(1, 2)), output_shape=output_rc.shape[1:],
                         name="reverse_complement_tformer_output_{n}".format(n=current_tformer))
    output_rc = revcomp_out(output_rc)
    out = concatenate([output_fwd, output_rc], axis=-1)
    return out


def add_siam_layernorm(inputs_fwd, inputs_rc, current_ln):
    input_shape = inputs_rc.shape
    if len(input_shape) != 3:
        raise ValueError("Intended for RC layers with 2D output."
                         "Expected dimension: 3, but got: " + str(len(input_shape)))
    out = concatenate([inputs_fwd, inputs_rc], axis=1)
    out = LayerNormalization(epsilon=1e-6)(out)
    split_shape = out.shape[1] // 2
    new_shape = [split_shape, input_shape[2]]
    fwd_out = Lambda(lambda x: x[:, :split_shape, :], output_shape=new_shape,
                     name="split_layernorm_fwd_output_{n}".format(n=current_ln))
    rc_out = Lambda(lambda x: x[:, split_shape:, :], output_shape=new_shape,
                    name="split_layernorm_rc_output1_{n}".format(n=current_ln))

    x_fwd = fwd_out(out)
    x_rc = rc_out(out)
    return x_fwd, x_rc


def get_positional_encoding(length, embed_dim):
    if embed_dim % 2 != 0:
        raise ValueError("Channel dimension of the input embedding for the transformer "
                         "with fixed position signal must be divisible by 2. Received: {}".format(embed_dim))
    position = K.arange(0, length, dtype=K.floatx())
    num_timescales = embed_dim // 2
    log_timescale_increment = np.log(10000) / (num_timescales - 1)
    inv_timescales = (K.exp(K.arange(num_timescales, dtype=K.floatx()) * (-log_timescale_increment)))
    scaled_time = K.expand_dims(position, 1) * K.expand_dims(inv_timescales, 0)
    signal = K.concatenate([K.sin(scaled_time), K.cos(scaled_time)], axis=1)
    return K.expand_dims(signal, axis=0)


class PositionEmbedding(Layer):
    def __init__(self, max_depth, seed, use_depth=True, fixed=True, **kwargs):
        self.max_depth = max_depth
        self.horizontal_position_embeddings = None
        self.vertical_position_embeddings = None
        self.seed = seed
        self.fixed = fixed
        self.use_depth = False if max_depth == 1 else use_depth
        self.initializer = tf.keras.initializers.RandomUniform(seed=self.seed)
        super().__init__(**kwargs)

    def get_config(self):
        config = super().get_config()
        config['max_depth'] = self.max_depth
        config['seed'] = self.seed
        config['use_depth'] = self.use_depth
        config['fixed'] = self.fixed
        return config

    def build(self, input_shape):
        seq_length, embed_dim = input_shape[-2:]
        if self.fixed:
            self.horizontal_position_embeddings = get_positional_encoding(seq_length, embed_dim)
            self.vertical_position_embeddings = self.horizontal_position_embeddings
        else:
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
            if self.fixed:
                K.expand_dims(self.vertical_position_embeddings[:, current_tformer, :], axis=1)
            else:
                out = out + self.vertical_position_embeddings[current_tformer, :]
        return out


get_custom_objects().update({
    'PositionEmbedding': PositionEmbedding,
})
