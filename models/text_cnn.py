from keras.layers import Embedding, Dense, Flatten, Input, concatenate
from keras.layers.convolutional import Conv1D
from keras.layers.pooling import MaxPool1D
from keras.models import Model


def TextCNN(sequence_len, vocab_size, embedding_dim, filter_sizes):
    # Input Layer
    input_layer = Input(shape=(sequence_len,))

    # Hideen Layer
    embedding_layer = Embedding(vocab_size, embedding_dim)(input_layer)

    pooled_outputs = []
    for filter_size in filter_sizes:
        x = Conv1D(embedding_dim, filter_size, activation='relu')(embedding_layer)
        x = MaxPool1D(pool_size=2)(x)
        pooled_outputs.append(x)

    merged = concatenate(pooled_outputs, axis=1)
    dense_layer = Flatten()(merged)

    # Output Layer
    output_layer = Dense(1, activation='sigmoid')(dense_layer)
    model = Model(inputs=input_layer, outputs=output_layer)

    return model

