import keras
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

from models.text_cnn import TextCNN
from utils import set_env, create_callbacks
from utils.data_helper import pre_processing
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
    train_dir = "../data/multi_train.csv"
    test_dir = "../data/multi_test.csv"
    model_dir = "./model_save"

    # HyperParameter
    epoch = 2
    batch = 256
    target_names = ['0', '1', '2', '3']

    # Flow
    print("0. Setting Environment")
    set_env()

    print("1. load data")
    train_x, train_y, test_x, test_y, val_x, val_y = load_data(train_dir, test_dir, len(target_names))

    print("2. pre processing")
    train_x, test_x, val_x, tokenizer = pre_processing(train_x, test_x, val_x)

    print("3. build model")
    model = TextCNN(
        sequence_len=train_x.shape[1],
        embedding_matrix=len(tokenizer.word_index) + 1,
        embedding_dim=300,
        filter_sizes=[3, 4, 5],
        flag="self_training",
        data_type="multi",
        category_num=len(target_names)
    )
    model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=['accuracy'])

    callbacks = create_callbacks(model_dir)
    model.fit(x=train_x, y=train_y, epochs=epoch, batch_size=batch, validation_data=(val_x, val_y), callbacks=callbacks)

    print("4. evaluation")
    evaluation = Evaluation(model, test_x, test_y)
    accuracy, cf_matrix, report = evaluation.eval_classification(data_type="multi")
    print("## Target Names : ", target_names)
    print("## Classification Report \n", report)
    print("## Confusion Matrix \n", cf_matrix)
    print("## Accuracy \n", accuracy)


if __name__ == "__main__":
    main()


