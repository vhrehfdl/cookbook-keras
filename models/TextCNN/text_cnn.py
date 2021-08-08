from keras.callbacks import ModelCheckpoint
from keras.layers import Embedding, Dense, Flatten, Input
from keras.layers import LSTM, Bidirectional, GlobalMaxPooling1D, GlobalAveragePooling1D
from keras.layers import SpatialDropout1D, add, concatenate
from keras.layers.convolutional import Conv1D
from keras.layers.pooling import MaxPool1D
from keras.models import Model
from keras.preprocessing import text, sequence

class TextCNN:
    def __init__(self, size, vocab_size):
        super(TextCNN, self).__init__()

        self.embedding_dim = 300
        self.num_filters = 128
        self.filter_sizes = [3, 4, 5]
        self.size = size
        self.vocab_size = vocab_size

    def build_model(self):
        input_layer = Input(shape=(self.size,))
        embedding_layer = Embedding(self.vocab_size, self.embedding_dim)(input_layer)

        pooled_outputs = []
        for filter_size in self.filter_sizes:
            x = Conv1D(self.embedding_dim, filter_size, activation='relu')(embedding_layer)
            x = MaxPool1D(pool_size=2)(x)
            pooled_outputs.append(x)

        merged = concatenate(pooled_outputs, axis=1)
        dense_layer = Flatten()(merged)

        output_layer = Dense(1, activation='sigmoid')(dense_layer)
        model = Model(inputs=input_layer, outputs=output_layer)

        return model

