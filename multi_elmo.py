import keras
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

from models.elmo import BasicElmo
from utils import set_env, create_callbacks
from utils.evaluation import Evaluation


def load_data(train_dir, test_dir, category_size):
    train = pd.read_csv(train_dir)
    test = pd.read_csv(test_dir)

    train, val = train_test_split(train, test_size=0.1, random_state=42)

    train_x, train_y = train["turn3"], train["label"]
    test_x, test_y = test["turn3"], test["label"]
    val_x, val_y = val["turn3"], val["label"]

    encoder = preprocessing.LabelEncoder()
    train_y = encoder.fit_transform(train_y)
    test_y = encoder.fit_transform(test_y)
    val_y = encoder.fit_transform(val_y)

    train_y = keras.utils.to_categorical(train_y, category_size)
    val_y = keras.utils.to_categorical(val_y, category_size)

    return train_x, train_y, test_x, test_y, val_x, val_y


def main():
    # Directory Setting
    train_dir = "./data/multi_train.csv"
    test_dir = "./data/multi_test.csv"
    model_dir = "./model_save"

    # HyperParameter
    epoch = 1
    batch = 128
    max_len = 50
    hidden_units = 64
    target_names = ['0', '1', '2', '3']

    # Flow
    print("0. Setting Environment")
    set_env()

    print("1. load data")
    train_x, train_y, test_x, test_y, val_x, val_y = load_data(train_dir, test_dir, len(target_names))

    print("2. pre processing")
    train_x, val_x, test_x = train_x.tolist(), val_x.tolist(), test_x.tolist()

    train_x = [' '.join(t.split()[0:max_len]) for t in train_x]
    train_x = np.array(train_x, dtype=object)[:, np.newaxis]

    val_x = [' '.join(t.split()[0:max_len]) for t in val_x]
    val_x = np.array(val_x, dtype=object)[:, np.newaxis]

    test_x = [' '.join(t.split()[0:max_len]) for t in test_x]
    test_x = np.array(test_x, dtype=object)[:, np.newaxis]

    print("3. build model")
    model = BasicElmo(
        hidden_units=hidden_units,
        data_type="multi",
        category_size=len(target_names)
    )
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    callbacks = create_callbacks(model_dir)
    model.fit(x=train_x, y=train_y, epochs=epoch, batch_size=batch, validation_data=(val_x, val_y), callbacks=callbacks)

    print("4. evaluation")
    evaluation = Evaluation(model, test_x, test_y)
    accuracy, cf_matrix, report = evaluation.eval_classification(data_type="multi")
    print("## Target Names : ", target_names)
    print("## Classification Report \n", report)
    print("## Confusion Matrix \n", cf_matrix)
    print("## Accuracy \n", accuracy)


if __name__ == '__main__':
    main()
