# partially inspired by
# https://keras.io/examples/nlp/text_classification_with_transformer/ (by Apoorv Nandan)
# and https://github.com/kpot/keras-transformer/blob/master/keras_transformer/position.py (by Kirill Mavreshko)
# implemented as functions since keras/tf had problems loading models with custom layers in custom layers
# (wrong weight order)
import tensorflow as tf
import numpy as np
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Dense, Lambda, LayerNormalization, Layer, Dropout,\
    Reshape, add, Activation, concatenate, Conv1D
from tensorflow.keras.utils import get_custom_objects


def separate_heads(input, qkv_name, seq_len, num_heads, projection_dim, current_tformer):
    out = Reshape((seq_len, num_heads, projection_dim))(input)
    perm = Lambda(lambda x: K.permute_dimensions(x, pattern=[0, 2, 1, 3]),
                  name="permute_attention_{qkv}_{n}".format(qkv=qkv_name, n=current_tformer))
    out = perm(out)
    return out


def get_attention(query, key, value, current_tformer, mode_name=None):
    score = tf.matmul(query, key, transpose_b=True)
    dim_key = key.shape[-1]
    if mode_name is None:
        layer_name = "scale_attention_{n}".format(n=current_tformer)
    else:
        layer_name = "scale_attention_{mode}_{n}".format(mode=mode_name, n=current_tformer)
    scale = Lambda(lambda x: x / np.sqrt(dim_key),
                   name=layer_name)
    scaled_score = scale(score)
    weights = Activation("softmax")(scaled_score)
    output = tf.matmul(weights, value)
    return output


def get_perf_attention(query, key, value, current_tformer, mode_name=None, activation="relu", kernel_epsilon=0.001):
    if mode_name is None:
        layer_name_postfix = "_{n}".format(n=current_tformer)
    else:
        layer_name_postfix = "_{mode}_{n}".format(mode=mode_name, n=current_tformer)

    add_eps = Lambda(lambda x: tf.add(x, kernel_epsilon), name="add_eps_{}".format(layer_name_postfix))
    query = add_eps(Activation(activation)(query))
    key = add_eps(Activation(activation)(key))
    value = add_eps(Activation(activation)(value))
    score_z = tf.matmul(key, value, transpose_a=True)
    # based on https://github.com/lucidrains/performer-pytorch/blob/main/performer_pytorch/performer_pytorch.py
    k_sum = tf.math.reduce_sum(key, axis=-2)
    invert = Lambda(lambda x: tf.truediv(tf.cast(1, x.dtype), x), name="d_inv_{}".format(layer_name_postfix))
    d_inv = tf.einsum('...nd,...d->...n', query, k_sum)
    d_inv = invert(d_inv)
    out = tf.einsum('...de,...nd,...n->...ne', score_z, query, d_inv)
    return out


def add_mhs_attention(inputs, embed_dim, num_heads, initializer, current_tformer, perf_dim=0):
    if embed_dim % num_heads != 0:
        raise ValueError(
            f"embedding dimension = {embed_dim} should be divisible by number of heads = {num_heads}"
        )

    if perf_dim > 0:
        projection_dim_m = perf_dim // num_heads
        q_init = tf.keras.initializers.Orthogonal(seed=2 * current_tformer)
        k_init = tf.keras.initializers.Orthogonal(seed=(2 * current_tformer) + 1)
        query_dense = Dense(perf_dim, kernel_initializer=q_init, trainable=False, use_bias=False,
                            name="random_projection_query_{}".format(current_tformer))
        key_dense = Dense(perf_dim, kernel_initializer=k_init, trainable=False, use_bias=False,
                          name="random_projection_key_{}".format(current_tformer))
        att_fction = get_perf_attention
    else:
        projection_dim_m = embed_dim // num_heads
        query_dense = Dense(embed_dim, kernel_initializer=initializer)
        key_dense = Dense(embed_dim, kernel_initializer=initializer)
        att_fction = get_attention
    projection_dim_v = embed_dim // num_heads
    value_dense = Dense(embed_dim, kernel_initializer=initializer)
    combine_heads = Dense(embed_dim, kernel_initializer=initializer)
    seq_len = inputs.shape[1]

    query = query_dense(inputs)
    key = key_dense(inputs)
    value = value_dense(inputs)
    query = separate_heads(query, "query", seq_len, num_heads, projection_dim_m, current_tformer)
    key = separate_heads(key, "key", seq_len, num_heads, projection_dim_m, current_tformer)
    value = separate_heads(value, "value", seq_len, num_heads, projection_dim_v, current_tformer)

    att_out = att_fction(query, key, value, current_tformer)

    perm = Lambda(lambda x: K.permute_dimensions(x, pattern=[0, 2, 1, 3]),
                  name="permute_attention_out_{n}".format(n=current_tformer))
    att_out = perm(att_out)
    att_out = Reshape((seq_len, embed_dim))(att_out)
    output = combine_heads(att_out)
    return output


def add_siam_mhs_attention(inputs_fwd, inputs_rc, embed_dim, num_heads, initializer, current_tformer, perf_dim=0):
    if embed_dim % num_heads != 0:
        raise ValueError(
            f"embedding dimension = {embed_dim} should be divisible by number of heads = {num_heads}"
        )

    if perf_dim > 0:
        projection_dim_m = perf_dim // num_heads
        q_init = tf.keras.initializers.Orthogonal(seed=2 * current_tformer)
        k_init = tf.keras.initializers.Orthogonal(seed=(2 * current_tformer) + 1)
        query_dense = Dense(perf_dim, kernel_initializer=q_init, trainable=False, use_bias=False,
                            name="random_projection_query_{}".format(current_tformer))
        key_dense = Dense(perf_dim, kernel_initializer=k_init, trainable=False, use_bias=False,
                          name="random_projection_key_{}".format(current_tformer))
        att_fction = get_perf_attention
    else:
        projection_dim_m = embed_dim // num_heads
        query_dense = Dense(embed_dim, kernel_initializer=initializer)
        key_dense = Dense(embed_dim, kernel_initializer=initializer)
        att_fction = get_attention
    projection_dim_v = embed_dim // num_heads
    value_dense = Dense(embed_dim, kernel_initializer=initializer)
    combine_heads = Dense(embed_dim, kernel_initializer=initializer)
    seq_len = inputs_fwd.shape[1]

    query_fwd = query_dense(inputs_fwd)
    key_fwd = key_dense(inputs_fwd)
    value_fwd = value_dense(inputs_fwd)
    query_rc = query_dense(inputs_rc)
    key_rc = key_dense(inputs_rc)
    value_rc = value_dense(inputs_rc)
    query_fwd = separate_heads(query_fwd, "query_fwd", seq_len, num_heads, projection_dim_m, current_tformer)
    key_fwd = separate_heads(key_fwd, "key_fwd", seq_len, num_heads, projection_dim_m, current_tformer)
    value_fwd = separate_heads(value_fwd, "value_fwd", seq_len, num_heads, projection_dim_v, current_tformer)
    query_rc = separate_heads(query_rc, "query_rc", seq_len, num_heads, projection_dim_m, current_tformer)
    key_rc = separate_heads(key_rc, "key_rc", seq_len, num_heads, projection_dim_m, current_tformer)
    value_rc = separate_heads(value_rc, "value_rc", seq_len, num_heads, projection_dim_v, current_tformer)

    att_out_fwd = att_fction(query_fwd, key_fwd, value_fwd, current_tformer, "fwd")
    att_out_rc = att_fction(query_rc, key_rc, value_rc, current_tformer, "rc")

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
                          current_tformer, seed, perf_dim=0, training=False):

    inputs = position_embedding(inputs, current_tformer=current_tformer)
    attn_output = add_mhs_attention(inputs, embed_dim, num_heads, initializer, current_tformer, perf_dim=perf_dim)

    layernorm1 = LayerNormalization(epsilon=1e-6)
    layernorm2 = LayerNormalization(epsilon=1e-6)

    if not np.isclose(dropout_rate, 0.0):
        attn_output = Dropout(dropout_rate, seed=seed)(attn_output, training=training)
    if inputs.shape[-1] != embed_dim:
        conv = Conv1D(filters=embed_dim, kernel_size=1, kernel_initializer=initializer,
                      name="conv_skip_tformer_{}".format(current_tformer))
        inputs = conv(inputs)
    out1 = layernorm1(add([inputs, attn_output]))
    ffn_output = Dense(ff_dim, activation="relu", kernel_initializer=initializer)(out1)
    ffn_output = Dense(embed_dim, kernel_initializer=initializer)(ffn_output)
    if not np.isclose(dropout_rate, 0.0):
        ffn_output = Dropout(dropout_rate, seed=seed)(ffn_output, training=training)
    return layernorm2(add([out1, ffn_output]))


def add_siam_transformer_block(inputs_fwd, inputs_rc, position_embedding, embed_dim, num_heads, ff_dim, dropout_rate,
                               initializer, current_tformer, seed, full_rc_att=False, full_rc_ffn=False,
                               perf_dim=0, training=False):

    inputs_fwd = position_embedding(inputs_fwd, current_tformer=current_tformer)
    inputs_rc = position_embedding(inputs_rc, current_tformer=current_tformer)
    attn_output_fwd, attn_output_rc = add_siam_mhs_attention(inputs_fwd, inputs_rc, embed_dim, num_heads, initializer,
                                                             current_tformer, perf_dim=perf_dim)

    if not np.isclose(dropout_rate, 0.0):
        attn_output_fwd = Dropout(dropout_rate, seed=seed)(attn_output_fwd, training=training)
        attn_output_rc = Dropout(dropout_rate, seed=seed)(attn_output_rc, training=training)
    if inputs_fwd.shape[-1] != embed_dim:
        conv = Conv1D(filters=embed_dim, kernel_size=1, kernel_initializer=initializer,
                      name="conv_skip_tformer_{}".format(current_tformer))
        inputs_fwd = conv(inputs_fwd)
        inputs_rc = conv(inputs_rc)
    if full_rc_att:
        out1_fwd, out1_rc = add_siam_layernorm(add([inputs_fwd, attn_output_fwd, attn_output_rc]),
                                               add([inputs_rc, attn_output_fwd, attn_output_rc]),
                                               current_ln=current_tformer * 2)
    else:
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
    if full_rc_ffn:
        output_fwd, output_rc = add_siam_layernorm(add([out1_fwd, ffn_output_fwd, ffn_output_rc]),
                                                   add([out1_rc, ffn_output_fwd, ffn_output_rc]),
                                                   current_ln=(current_tformer*2)+1)
    else:
        output_fwd, output_rc = add_siam_layernorm(add([out1_fwd, ffn_output_fwd]), add([out1_rc, ffn_output_rc]),
                                                   current_ln=(current_tformer*2)+1)
    return output_fwd, output_rc


def add_rc_transformer_block(inputs, embed_dim, position_embedding, num_heads, ff_dim, dropout_rate, initializer,
                             current_tformer, seed, keep_edim_fction=None, full_rc_att=False, full_rc_ffn=False,
                             perf_dim=0, training=False):

    revcomp_in = Lambda(lambda x: K.reverse(x, axes=(1, 2)), output_shape=inputs.shape[1:],
                        name="reverse_complement_tformer_input_{n}".format(n=current_tformer))
    inputs_rc = revcomp_in(inputs)
    output_fwd, output_rc = add_siam_transformer_block(inputs, inputs_rc, position_embedding, embed_dim, num_heads,
                                                       ff_dim, dropout_rate, initializer, current_tformer, seed,
                                                       full_rc_att, full_rc_ffn, perf_dim=perf_dim, training=training)

    revcomp_out = Lambda(lambda x: K.reverse(x, axes=(1, 2)), output_shape=output_rc.shape[1:],
                         name="reverse_complement_tformer_output_{n}".format(n=current_tformer))
    output_rc = revcomp_out(output_rc)
    out = concatenate([output_fwd, output_rc], axis=-1)
    if keep_edim_fction is not None:
        out = keep_edim_fction(out)

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


# def squash(x):
#     squasher = K.expand_dims(K.arange(x.shape[-1], dtype=x.dtype) + 1, axis=0)
#     x = tf.matmul(x, squasher, transpose_b=True)
#     x = tf.reduce_sum(x, axis=-1)
#     return x
#
#
# def add_squash_back(x, mode_name=None):
#     if mode_name is None:
#         layer_name = "squashback"
#     else:
#         layer_name = "squashback_{}".format(mode_name)
#     sq = Lambda(lambda _x: squash(_x), name=layer_name)
#     return sq(x)
#
#
# def add_embedding(x, input_dim, output_dim, seed):
#     x = add_squash_back(x)
#     init = tf.keras.initializers.RandomUniform(seed=seed)
#     emb = Embedding(input_dim=input_dim, output_dim=output_dim, embeddings_initializer=init)
#     return emb(x)
#
#
# def add_siam_embedding(x_fwd, x_rc, input_dim, output_dim, seed):
#     x_fwd = add_squash_back(x_fwd, "fwd")
#     x_rc = add_squash_back(x_rc, "rc")
#     init = tf.keras.initializers.RandomUniform(seed=seed)
#     emb = Embedding(input_dim=input_dim, output_dim=output_dim, embeddings_initializer=init)
#     return emb(x_fwd), emb(x_rc)
#
#
# def add_rc_embedding(x, input_dim, output_dim, seed):
#     revcomp_in = Lambda(lambda _x: K.reverse(_x, axes=(1, 2)), output_shape=x.shape[1:],
#                         name="reverse_complement_embed_input")
#     x_rc = revcomp_in(x)
#     x_fwd, x_rc = add_siam_embedding(x, x_rc, input_dim, output_dim, seed)
#     revcomp_out = Lambda(lambda _x: K.reverse(_x, axes=(1, 2)), output_shape=x_rc.shape[1:],
#                          name="reverse_complement_embed_output")
#     x_rc = revcomp_out(x_rc)
#     out = concatenate([x_fwd, x_rc], axis=-1)
#     return out


def get_position_encoding(length, embed_dim, rc_folds=1, dtype='float32'):
    if embed_dim % 2 != 0:
        raise ValueError("Channel dimension of the input embedding for the transformer "
                         "with fixed position signal must be divisible by 2. Received: {}".format(embed_dim))
    position = K.arange(0, length, dtype=dtype)
    num_timescales = embed_dim // (2**rc_folds)
    log_timescale_increment = np.log(10000) / (num_timescales - 1)
    inv_timescales = (K.exp(K.arange(num_timescales, dtype=dtype) * (-log_timescale_increment)))
    scaled_time = K.expand_dims(position, 1) * K.expand_dims(inv_timescales, 0)
    # This follows the Tensor2Tensor library implementation
    # (https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/layers/common_attention.py)
    # Note that this slightly differs from the ordering in the Attention Is All You Need paper
    # The difference doesn't really matter.
    # See a discussion here: https://github.com/tensorflow/tensor2tensor/pull/177
    signal = K.concatenate([K.sin(scaled_time), K.cos(scaled_time)], axis=1)
    for i in range(1, rc_folds):
        signal_rc = K.reverse(signal, axes=(0, 1))
        signal = K.concatenate([signal, signal_rc], axis=1)

    return K.expand_dims(signal, axis=0)


class PositionEmbedding(Layer):
    def __init__(self, max_depth, seed, use_depth=False, growing_rc=False, **kwargs):
        self.max_depth = max_depth
        self.horizontal_position_embeddings = None
        self.vertical_position_embeddings = None
        self.seed = seed
        self.growing_rc = growing_rc
        self.seq_length = None
        self.embed_dim = None
        self.use_depth = False if max_depth == 1 else use_depth
        self.initializer = tf.keras.initializers.RandomUniform(seed=self.seed)
        super().__init__(**kwargs)

    def get_config(self):
        config = super().get_config()
        config['max_depth'] = self.max_depth
        config['seed'] = self.seed
        config['use_depth'] = self.use_depth
        config['growing_rc'] = self.growing_rc
        return config

    def build(self, input_shape):
        self.seq_length, self.embed_dim = input_shape[-2:]
        super().build(input_shape)

    def call(self, inputs, **kwargs):
        current_tformer = kwargs.get('current_tformer')
        rc_folds = current_tformer + 1 if self.growing_rc else 1
        position_encoding = get_position_encoding(self.seq_length, self.embed_dim, rc_folds=rc_folds,
                                                  dtype=inputs.dtype)
        out = inputs + position_encoding
        if self.use_depth:
            out = out + K.expand_dims(position_encoding[:, current_tformer, :], axis=1)
        return out


get_custom_objects().update({
    'PositionEmbedding': PositionEmbedding,
})
