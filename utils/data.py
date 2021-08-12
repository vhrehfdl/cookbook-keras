import pandas as pd
from keras.preprocessing import text, sequence
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

class DataLoading:
    def __init__(self, train_dir, test_dir, test_size):
        train = pd.read_csv(train_dir)
        test = pd.read_csv(test_dir)

        train, val = train_test_split(train, test_size=test_size, random_state=42)

        self.train_x, self.train_y = train["text"], train["label"]
        self.val_x, self.val_y = val["text"], val["label"]
        self.test_x, self.test_y = test["text"], test["label"]

    def load_data_binary(self):
        encoder = preprocessing.LabelEncoder()
        train_y = encoder.fit_transform(self.train_y)
        val_y = encoder.fit_transform(self.val_y)
        test_y = encoder.fit_transform(self.test_y)

        target_names = encoder.classes_

        return self.train_x, train_y, self.test_x, test_y, self.val_x, val_y, target_names


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

    total_list = list(train_x) + list(test_x) + list(val_x)
    max_len = max([len(total_list[i]) for i in range(0, len(total_list))])

    train_x = sequence.pad_sequences(train_x, maxlen=max_len, padding='post')
    test_x = sequence.pad_sequences(test_x, maxlen=max_len, padding='post')
    val_x = sequence.pad_sequences(val_x, maxlen=max_len, padding='post')

    return train_x, test_x, val_x, tokenizer