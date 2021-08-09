import pandas as pd

import keras
from keras.callbacks import ModelCheckpoint
from keras.preprocessing import text, sequence
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

from utils import set_env
from utils.evaluation import Evaluation
from models.text_cnn import TextCNN


# load data from csv file.
def load_data(train_dir, test_dir):
    train = pd.read_csv(train_dir)
    test = pd.read_csv(test_dir)

    train, val = train_test_split(train, test_size=0.1, random_state=42)

    train_x, train_y = train["text"], train["label"]
    test_x, test_y = test["text"], test["label"]
    val_x, val_y = val["text"], val["label"]

    encoder = preprocessing.LabelEncoder()
    train_y = encoder.fit_transform(train_y)
    test_y = encoder.fit_transform(test_y)
    val_y = encoder.fit_transform(val_y)

    target_names = encoder.classes_

    return train_x, train_y, test_x, test_y, val_x, val_y, target_names


# convert Text data to vector.
def pre_processing(train_x, test_x, val_x):
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


def binary_evaluate(model, test_x, test_y):
    prediction = model.predict(test_x)
    y_pred = (prediction > 0.5)

    accuracy = accuracy_score(test_y, y_pred)
    cf_matrix = confusion_matrix(test_y, y_pred)
    report = classification_report(test_y, y_pred)

    return accuracy, cf_matrix, report


def create_callbacks(model_dir):
    checkpoint_callback = ModelCheckpoint(filepath=model_dir + "/model-weights.{epoch:02d}-{val_acc:.6f}.hdf5", monitor='val_acc', verbose=1, save_best_only=True)
    return [checkpoint_callback]


def main():
    # Directory Setting
    train_dir = "./data/binary_train.csv"
    test_dir = "./data/binary_test.csv"
    model_dir = "./model_save"

    # HyperParameter
    embedding_dim = 300
    filter_sizes = [3, 4, 5]
    epoch = 1
    batch = 256


    # Flow
    set_env()

    print("1. load data")
    train_x, train_y, test_x, test_y, val_x, val_y, target_names = load_data(train_dir, test_dir)
    
    print("2. pre processing & text to vector")
    train_x, test_x, val_x, tokenizer = pre_processing(train_x, test_x, val_x)
    sequence_len = train_x.shape[1]
    vocab_size = len(tokenizer.word_index) + 1

    print("3. build model")
    model = TextCNN(sequence_len, vocab_size, embedding_dim, filter_sizes)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    callbacks = create_callbacks(model_dir)
    model.fit(x=train_x, y=train_y, epochs=epoch, batch_size=batch, validation_data=(val_x, val_y), callbacks=callbacks)

    print("4. evaluation")
    evaluation = Evaluation(model, test_x, test_y)
    accuracy, cf_matrix, report = evaluation.eval_classification(data_type="binary")
    print("## Target Names : ", target_names)
    print("## Classification Report \n", report)
    print("## Confusion Matrix \n", cf_matrix)
    print("## Accuracy \n", accuracy)


if __name__ == '__main__':
    main()
