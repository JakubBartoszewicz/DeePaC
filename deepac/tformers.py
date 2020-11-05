# inspired by https://keras.io/examples/nlp/text_classification_with_transformer/ by Apoorv Nandan
import tensorflow as tf
import numpy as np
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Dense, Lambda, LayerNormalization, Embedding, Dropout, Reshape, add, Activation, Dot
from tensorflow.keras.activations import softmax


def add_mhs_attention(inputs, embed_dim, num_heads, initializer='glorot_uniform'):
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

    def separate_heads(input):
        out = Reshape((seq_len, num_heads, projection_dim))(input)
        perm = Lambda(lambda x: K.permute_dimensions(x, pattern=[0, 2, 1, 3]))
        out = perm(out)
        return out

    def get_attention(query, key, value):
        perm = Lambda(lambda x: K.permute_dimensions(x, [0, 1, 3, 2]))
        k_t = perm(key)
        score = Reshape((num_heads, seq_len, seq_len))(Dot(axes=(3, 2))([query, k_t]))
        dim_key = key.shape[-1]
        scale = Lambda(lambda x: x / np.sqrt(dim_key))
        scaled_score = scale(score)
        weights = Activation(softmax)(scaled_score)
        output = Reshape((num_heads, seq_len, projection_dim))(Dot(axes=(2, 2))([weights, value]))
        return output, weights

    query = query_dense(inputs)  # (batch_size, seq_len, embed_dim)
    key = key_dense(inputs)  # (batch_size, seq_len, embed_dim)
    value = value_dense(inputs)  # (batch_size, seq_len, embed_dim)
    query = separate_heads(query)  # (batch_size, num_heads, seq_len, projection_dim)
    key = separate_heads(key)  # (batch_size, num_heads, seq_len, projection_dim)
    value = separate_heads(value)  # (batch_size, num_heads, seq_len, projection_dim)
    att_out, weights = get_attention(query, key, value)
    perm = Lambda(lambda x: K.permute_dimensions(x, pattern=[0, 2, 1, 3]))
    att_out = perm(att_out)  # (batch_size, seq_len, num_heads, projection_dim)
    att_out = Reshape((seq_len, embed_dim))(att_out)
    # (batch_size, seq_len, embed_dim)
    output = combine_heads(att_out)  # (batch_size, seq_len, embed_dim)
    return output


def add_transformer_block(inputs, embed_dim, num_heads, ff_dim, dropout_rate, initializer, training=False):
    attn_output = add_mhs_attention(inputs, embed_dim, num_heads, initializer)

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


def add_token_position_embedding(inputs, maxlen, vocab_size, embed_dim, token_identity=True, training=False):
    token_emb = Embedding(input_dim=vocab_size, output_dim=embed_dim) if not token_identity else None
    pos_emb = Embedding(input_dim=maxlen, output_dim=embed_dim)

    positions = K.arange(start=0, stop=maxlen, step=1)
    positions = pos_emb(positions)
    if not token_identity:
        inputs = token_emb(inputs)
    out = inputs + positions
    return out

