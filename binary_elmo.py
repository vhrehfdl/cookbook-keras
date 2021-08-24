import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from utils import set_env, create_callbacks
from utils.data import pre_processing
from models.basic_elmo import BasicElmo
from utils.evaluation import Evaluation


def load_data(train_dir, test_dir):
    train = pd.read_csv(train_dir)
    test = pd.read_csv(test_dir)

    train, val = train_test_split(train, test_size=0.1, random_state=42)

    train_x, train_y = train["text"], train["label"]
    test_x, test_y = test["text"], test["label"]
    val_x, val_y = val["text"], val["label"]

    return train_x, train_y, test_x, test_y, val_x, val_y


def main():
    # Directory Setting
    train_dir = "./data/binary_train.csv"
    test_dir = "./data/binary_test.csv"
    model_dir = "./model_save"


    # HyperParameter
    max_len = 50
    epoch = 2
    batch = 256
    hidden_units = 256



    # Flow
    print("0. Setting Environment")
    set_env()


    print("1. load data")
    train_x, train_y, test_x, test_y, val_x, val_y = load_data(train_dir, test_dir)


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
        hidden_units = hidden_units
    )

    callbacks = create_callbacks(model_dir)
    model.fit(x=train_x, y=train_y, epochs=3, batch_size=128, validation_data=(val_x, val_y), callbacks=callbacks)


    print("4. evaluation")
    evaluation = Evaluation(model, test_x, test_y)
    accuracy, cf_matrix, report = evaluation.eval_classification(data_type="binary")
    print("## Classification Report \n", report)
    print("## Confusion Matrix \n", cf_matrix)
    print("## Accuracy \n", accuracy)



if __name__ == '__main__':
    main()
