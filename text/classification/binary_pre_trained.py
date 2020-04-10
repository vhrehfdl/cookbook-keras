import os
import pickle

import numpy as np
import pandas as pd
import tensorflow as tf
from keras.callbacks import ModelCheckpoint
from keras.layers import Bidirectional, GlobalMaxPooling1D, GlobalAveragePooling1D, LSTM
from keras.layers import Embedding, Dense, Flatten, Input
from keras.layers import add, concatenate
from keras.layers.convolutional import Conv1D
from keras.layers.pooling import MaxPool1D
from keras.models import Model
from keras.preprocessing import text, sequence
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


# gpu setting.
def set_env():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.5
    session = tf.Session(config=config)
    session


# load data from csv file.
def load_data(train_dir, test_dir):
    train = pd.read_csv(train_dir)
    test = pd.read_csv(test_dir)

    train, val = train_test_split(train, test_size=0.1, random_state=42)

    train_x, train_y = train["text"], train["label"]
    test_x, test_y = test["text"], test["label"]
    val_x, val_y = val["text"], val["label"]

    return train_x, train_y, test_x, test_y, val_x, val_y


# convert text data to vector.
def data_preprocissing(train_x, test_x, val_x):
    CHARS_TO_REMOVE = r'!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n“”’\'∞θ÷α•à−β∅³π‘₹´°£€\×™√²—'

    train_x = train_x.tolist()
    test_x = test_x.tolist()
    val_x = val_x.tolist()

    tokenizer = text.Tokenizer(filters=CHARS_TO_REMOVE)
    tokenizer.fit_on_texts(train_x + test_x + val_x)  # Make dictionary

    # Text match to dictionary.
    train_x = tokenizer.texts_to_sequences(train_x)
    test_x = tokenizer.texts_to_sequences(test_x)
    val_x = tokenizer.texts_to_sequences(val_x)

    temp_list = []
    total_list = list(train_x) + list(test_x) + list(val_x)

    for i in range(0, len(total_list)):
        temp_list.append(len(total_list[i]))

    max_len = max(temp_list)

    train_x = sequence.pad_sequences(train_x, maxlen=max_len, padding='post')
    test_x = sequence.pad_sequences(test_x, maxlen=max_len, padding='post')
    val_x = sequence.pad_sequences(val_x, maxlen=max_len, padding='post')

    return train_x, test_x, val_x, tokenizer


def get_coefs(word, *arr):
    return word, np.asarray(arr, dtype='float32')


def load_embeddings(path):
    with open(path, encoding="utf-8") as f:
        return dict(get_coefs(*line.strip().split(' ')) for line in f)


# Pre-trained embedding match to my dataset.
def text_to_vector(word_index, path, word_dimension):
    # If you change your embedding.pickle file, you must make new embedding.pickle file.
    if os.path.isfile("embedding_binary.pickle"):
        with open("embedding_binary.pickle", 'rb') as rotten_file:
            embedding_matrix = pickle.load(rotten_file)

    else:
        embedding_index = load_embeddings(path)
        embedding_matrix = np.zeros((len(word_index) + 1, 300))
        for word, i in word_index.items():
            try:
                embedding_matrix[i] = embedding_index[word]
            except KeyError:
                pass

        with open("embedding_binary.pickle", 'wb') as handle:
            pickle.dump(embedding_matrix, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return embedding_matrix


def build_model_basic(size, embedding_matrix):
    ### Hyper Parameter
    hidden_units = 64

    ### Model Architecture
    input_layer = Input(shape=(size,))

    embedding_layer = Embedding(*embedding_matrix.shape, weights=[embedding_matrix], trainable=False)(input_layer)

    dense_layer = Flatten()(embedding_layer)
    hidden_layer = Dense(hidden_units, activation='relu')(dense_layer)
    output_layer = Dense(1, activation='sigmoid')(hidden_layer)

    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()

    return model


# BI_LSTM
def build_model_lstm(size, embedding_matrix):
    ### Hyper Parameter
    lstm_units = 128
    hidden_units = 512

    ### Model Architecture
    input_layer = Input(shape=(size,))

    embedding_layer = Embedding(*embedding_matrix.shape, weights=[embedding_matrix], trainable=False)(input_layer)

    lstm_layer = Bidirectional(LSTM(lstm_units, return_sequences=True))(embedding_layer)
    lstm_layer = Bidirectional(LSTM(lstm_units, return_sequences=True))(lstm_layer)

    hidden_layer = concatenate([GlobalMaxPooling1D()(lstm_layer), GlobalAveragePooling1D()(lstm_layer)])
    hidden_layer = add([hidden_layer, Dense(hidden_units, activation='relu')(hidden_layer)])
    hidden_layer = add([hidden_layer, Dense(hidden_units, activation='relu')(hidden_layer)])

    output_layer = Dense(1, activation='sigmoid')(hidden_layer)

    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()

    return model


def build_model_cnn(size, embedding_matrix):
    ### Hyper Parameter
    embedding_dim = 300
    num_filters = 128
    filter_sizes = [3, 4, 5]

    ### Model Architecture
    input_layer = Input(shape=(size,))

    embedding_layer = Embedding(*embedding_matrix.shape, weights=[embedding_matrix], trainable=False)(input_layer)

    pooled_outputs = []
    for filter_size in filter_sizes:
        x = Conv1D(num_filters, filter_size, activation='relu')(embedding_layer)
        x = MaxPool1D(pool_size=2)(x)
        pooled_outputs.append(x)

    merged = concatenate(pooled_outputs, axis=1)
    dense_layer = Flatten()(merged)

    output_layer = Dense(1, activation='sigmoid')(dense_layer)

    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()

    return model


def evaluate(model, test_x, test_y):
    prediction = model.predict(test_x)
    y_pred = (prediction > 0.5)

    accuracy = accuracy_score(test_y, y_pred)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    print(classification_report(test_y, y_pred, target_names=["0", "1"]))


def create_callbacks(model_dir):
    checkpoint_callback = ModelCheckpoint(filepath=model_dir + "/model-weights.{epoch:02d}-{val_acc:.6f}.hdf5", monitor='val_acc', verbose=1, save_best_only=True)
    return [checkpoint_callback]


def main():
    ### Directory Setting.
    base_dir = "../.."

    train_dir = base_dir + "/Data/binary_train_data.csv"
    test_dir = base_dir + "/Data/binary_test_data.csv"

    model_dir = base_dir + "/Model"
    embedding_dir = base_dir + "/Data/glove.840B.300d.txt"


    ### Flow
    set_env()

    train_x, train_y, test_x, test_y, val_x, val_y = load_data(train_dir, test_dir)

    train_x, test_x, val_x, tokenizer = data_preprocissing(train_x, test_x, val_x)

    embedding_matrix = text_to_vector(tokenizer.word_index, embedding_dir, word_dimension=300)

    # model = build_model_basic(train_x.shape[1], embedding_matrix)
    # model = build_model_lstm(train_x.shape[1], embedding_matrix)
    model = build_model_cnn(train_x.shape[1], embedding_matrix)

    callbacks = create_callbacks(model_dir)
    model.fit(x=train_x, y=train_y, epochs=3, batch_size=128, validation_data=(val_x, val_y), callbacks=callbacks)

    evaluate(model, test_x, test_y)


if __name__ == '__main__':
    main()
