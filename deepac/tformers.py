# based on https://keras.io/examples/nlp/text_classification_with_transformer/ by Apoorv Nandan
import tensorflow as tf
import numpy as np
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Dense, Lambda, LayerNormalization, Layer, Embedding, Dropout, Reshape


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
    #batch_size = tf.shape(inputs)[0]
    #batch_size = K.shape(inputs)[0]
    seq_len = inputs.shape[1]

    def separate_heads(input):
        out = Reshape((seq_len, num_heads, projection_dim))(input)
        return Lambda(lambda x: K.permute_dimensions(x, pattern=[0, 2, 1, 3]))(out)

    def get_attention(query, key, value):
        k_t = Lambda(lambda x: K.permute_dimensions(x, [0, 1, 3, 2]))(key)
        score = Reshape((num_heads, seq_len, seq_len))(K.batch_dot(query, k_t, axes=(3, 2)))
        #score = tf.matmul(query, key, transpose_b=True)
        #dim_key = K.cast(K.shape(key)[-1], tf.float32)
        dim_key = key.shape[-1]
        scaled_score = Lambda(lambda x: x / np.sqrt(dim_key))(score)
        weights = Lambda(lambda x: K.softmax(x, axis=-1))(scaled_score)
        output = Reshape((num_heads, seq_len, projection_dim))(K.batch_dot(weights, value, axes=(2, 2)))
        #output = tf.matmul(weights, value)
        return output, weights

    #l_sep_heads = Lambda(lambda x: separate_heads(x))
    #l_attention = Lambda(lambda qkv: get_attention(qkv[0], qkv[1], qkv[2]))

    query = query_dense(inputs)  # (batch_size, seq_len, embed_dim)
    key = key_dense(inputs)  # (batch_size, seq_len, embed_dim)
    value = value_dense(inputs)  # (batch_size, seq_len, embed_dim)
    query = separate_heads(query)  # (batch_size, num_heads, seq_len, projection_dim)
    key = separate_heads(key)  # (batch_size, num_heads, seq_len, projection_dim)
    value = separate_heads(value)  # (batch_size, num_heads, seq_len, projection_dim)
    att_out, weights = get_attention(query, key, value)
    # att_out = tf.transpose(
    #     att_out, perm=[0, 2, 1, 3]
    # )  # (batch_size, seq_len, num_heads, projection_dim)
    att_out = Lambda(lambda x: K.permute_dimensions(x, pattern=[0, 2, 1, 3]))(att_out)  # (batch_size, seq_len, num_heads, projection_dim)
    # Explicitly declare shape
    #concat_attention = Lambda(lambda x: tf.reshape(x, (batch_size, seq_len, embed_dim)))
    #, output_shape=(inputs.shape[1], embed_dim)
    att_out = Reshape((seq_len, embed_dim))(att_out)
    #att_out = tf.reshape(attention, (batch_size, -1, embed_dim))
    #att_out = tf.reshape(att_out, (batch_size, seq_len, embed_dim))
    # (batch_size, seq_len, embed_dim)
    output = combine_heads(att_out)  # (batch_size, seq_len, embed_dim)
    return output


def add_transformer_block(inputs, embed_dim, num_heads, ff_dim, dropout_rate, initializer, training=False):
    #att = MultiHeadSelfAttention(embed_dim, num_heads, initializer)
    attn_output = add_mhs_attention(inputs, embed_dim, num_heads, initializer)

    layernorm1 = LayerNormalization(epsilon=1e-6)
    layernorm2 = LayerNormalization(epsilon=1e-6)
    dropout1 = Dropout(dropout_rate)
    dropout2 = Dropout(dropout_rate)

    #attn_output = att(inputs)
    if not np.isclose(dropout_rate, 0.0):
        attn_output = dropout1(attn_output, training=training)
    out1 = layernorm1(inputs + attn_output)
    ffn_output = Dense(ff_dim, activation="relu", kernel_initializer=initializer)(out1)
    ffn_output = Dense(embed_dim, kernel_initializer=initializer)(ffn_output)
    if not np.isclose(dropout_rate, 0.0):
        ffn_output = dropout2(ffn_output, training=training)
    return layernorm2(out1 + ffn_output)


class TokenAndPositionEmbedding(Layer):
    def __init__(self, maxlen, vocab_size, embed_dim, token_identity=True, **kwargs):
        super(TokenAndPositionEmbedding, self).__init__(kwargs)
        self.maxlen = maxlen
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.token_identity = token_identity
        self.token_emb = Embedding(input_dim=vocab_size, output_dim=embed_dim) if not token_identity else None
        self.pos_emb = Embedding(input_dim=maxlen, output_dim=embed_dim)

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
        positions = K.arange(start=0, stop=self.maxlen, step=1)
        positions = self.pos_emb(positions)
        if not self.token_identity:
            inputs = self.token_emb(inputs)
        return inputs + positions

