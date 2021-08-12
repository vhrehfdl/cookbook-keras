import os
import pickle
import tensorflow as tf

import numpy as np
from utils import set_env, create_callbacks
from utils.evaluation import Evaluation
from utils.data import DataLoading, pre_processing
from models.text_cnn import TextCNN


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


def main():
    # Directory Setting
    train_dir = "./data/binary_train.csv"
    test_dir = "./data/binary_test.csv"
    model_dir = "./model_save"
    embedding_dir = "./glove.6B.300d.txt"

    # HyperParameter
    embedding_dim = 300
    filter_sizes = [3, 4, 5]
    epoch = 2
    batch = 256


    # Flow
    set_env()

    print("1. load data")
    data_loading = DataLoading(train_dir, test_dir, test_size=0.1)
    train_x, train_y, test_x, test_y, val_x, val_y, target_names = data_loading.load_data_binary()
    
    print("2. pre processing")
    train_x, test_x, val_x, tokenizer = pre_processing(train_x, test_x, val_x)
    sequence_len = train_x.shape[1]
    vocab_size = len(tokenizer.word_index) + 1 

    print("3. text to vector")
    embedding_matrix = text_to_vector(tokenizer.word_index, embedding_dir, word_dimension=300)
    
    print("4. build model")
    model = TextCNN(sequence_len, embedding_matrix, embedding_dim, filter_sizes, flag="pre_trained")
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    callbacks = create_callbacks(model_dir)
    model.fit(x=train_x, y=train_y, epochs=epoch, batch_size=batch, validation_data=(val_x, val_y), callbacks=callbacks)

    print("5. evaluation")
    evaluation = Evaluation(model, test_x, test_y)
    accuracy, cf_matrix, report = evaluation.eval_classification(data_type="binary")
    print("## Target Names : ", target_names)
    print("## Classification Report \n", report)
    print("## Confusion Matrix \n", cf_matrix)
    print("## Accuracy \n", accuracy)


if __name__ == '__main__':
    main()
