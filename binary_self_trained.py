from utils import set_env, create_callbacks
from utils.evaluation import Evaluation
from utils.data import DataLoading, pre_processing
from models.text_cnn import TextCNN


def main():
    # Directory Setting
    train_dir = "./data/binary_train.csv"
    test_dir = "./data/binary_test.csv"
    model_dir = "./model_save"

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

    print("3. build model")
    model = TextCNN(sequence_len, vocab_size, embedding_dim, filter_sizes, flag="self")
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
