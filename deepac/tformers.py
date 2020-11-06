# inspired by https://keras.io/examples/nlp/text_classification_with_transformer/ by Apoorv Nandan
# implemented as functions as keras/tf had problems loading models with custom layers in custom layers
# (wrong weight order)
import tensorflow as tf
import numpy as np
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Dense, Lambda, LayerNormalization, Embedding, Dropout, Reshape, add, Activation, Dot
from tensorflow.keras.activations import softmax


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

    # x.shape = [batch_size, seq_len, embedding_dim]
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

    query = query_dense(inputs)  # (batch_size, seq_len, embed_dim)
    key = key_dense(inputs)  # (batch_size, seq_len, embed_dim)
    value = value_dense(inputs)  # (batch_size, seq_len, embed_dim)
    query = separate_heads(query, "query")  # (batch_size, num_heads, seq_len, projection_dim)
    key = separate_heads(key, "key")  # (batch_size, num_heads, seq_len, projection_dim)
    value = separate_heads(value, "value")  # (batch_size, num_heads, seq_len, projection_dim)
    att_out, weights = get_attention(query, key, value)
    perm = Lambda(lambda x: K.permute_dimensions(x, pattern=[0, 2, 1, 3]),
                  name="permute_attention_out_{n}".format(n=current_tformer))
    att_out = perm(att_out)  # (batch_size, seq_len, num_heads, projection_dim)
    att_out = Reshape((seq_len, embed_dim))(att_out)
    # (batch_size, seq_len, embed_dim)
    output = combine_heads(att_out)  # (batch_size, seq_len, embed_dim)
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


def add_position_embedding(inputs, max_len, max_depth, embed_dim, current_tformer):
    horiz_emb = Embedding(input_dim=max_len, output_dim=embed_dim)
    horiz_positions = K.arange(start=0, stop=max_len, step=1)
    horiz_positions_emb = horiz_emb(horiz_positions)
    vertical_emb = Embedding(input_dim=max_len, output_dim=embed_dim)
    vertical_positions = K.arange(start=0, stop=max_depth, step=1)
    vertical_positions_emb = vertical_emb(vertical_positions)
    out = inputs + horiz_positions_emb
    out = out + vertical_positions_emb[current_tformer]
    return out

