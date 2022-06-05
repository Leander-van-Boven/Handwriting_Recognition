import itertools
import os
import string
from pathlib import Path

import PIL
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from tensorflow.python.keras.callbacks import EarlyStopping, TensorBoard
from tqdm import tqdm

from src.iam.model_architecture import get_model, compile_model
from src.utils.csv_writer import CSVWriter


# globals
batch_size = 64
epochs = 20
num_models = 3
num_images = (1536, 192, 192)
WINDOW_SIZE = 64

# paths
parent_dir = Path('../../data/iam/IMData_Split').resolve()
TRAIN_PATH = os.path.join(parent_dir, 'Train')
TEST_PATH = os.path.join(parent_dir, 'Test')
VALIDATION_PATH = os.path.join(parent_dir, 'Validation')


def read_in_data():
    """
    Reads in the data from the directories and returns the data and labels.
    :return: train_ds, train_labels, test_ds, test_labels, validation_ds, validation_labels, class_names
    """

    # initiate variables
    letters = []

    train_ds = []
    test_ds = []
    validation_ds = []

    train_labels = []
    test_labels = []
    validation_labels = []

    for letter in os.listdir(TRAIN_PATH):
        letters.append(letter)

    # labels are one hot encoded
    ascii_characters = string.ascii_letters + string.digits
    def hex_char_to_one_hot(hex_char):
        zeros = np.zeros(len(ascii_characters) + 1)
        if hex_char == 'ZBlank':
            zeros[-1] = 1
        else:
            zeros[ascii_characters.index(chr(int(hex_char, 16)))] = 1
        return zeros
    one_hot_encoded_labels = {
        hex_char: hex_char_to_one_hot(hex_char) for hex_char in letters
    }

    enumerator = 0
    for letter in tqdm(os.listdir(TRAIN_PATH), desc='Reading in Training Data'):
        image_folder = os.path.join(TRAIN_PATH, letter)
        for counter, im_file in enumerate(os.listdir(image_folder)):
            image = PIL.Image.open(os.path.join(image_folder, im_file))
            image_arr = np.array(image)
            # Get WINDOW_SIZE wide window and normalize
            padding = (image_arr.shape[1] - WINDOW_SIZE) // 2
            image_arr = image_arr[:, padding:padding+WINDOW_SIZE] // 255
            assert image_arr.shape[1] == WINDOW_SIZE
            # Invert image to make blackground white
            image_arr = np.where(image_arr == 0, 1, 0)
            train_ds.append(image_arr)
            train_labels.append(one_hot_encoded_labels[letter])
        enumerator += 1

    enumerator = 0
    for letter in tqdm(os.listdir(TEST_PATH), desc='Reading in Test Data'):
        image_folder = os.path.join(TEST_PATH, letter)
        for im_file in os.listdir(image_folder):
            image = PIL.Image.open(os.path.join(image_folder, im_file))
            image_arr = np.array(image)
            # Get WINDOW_SIZE wide window and normalize
            padding = (image_arr.shape[1] - WINDOW_SIZE) // 2
            image_arr = image_arr[:, padding:padding+WINDOW_SIZE] // 255
            assert image_arr.shape[1] == WINDOW_SIZE
            # Invert image to make blackground white
            image_arr = np.where(image_arr == 0, 1, 0)
            test_ds.append(image_arr)
            test_labels.append(one_hot_encoded_labels[letter])
        enumerator += 1

    enumerator = 0
    for letter in tqdm(os.listdir(VALIDATION_PATH), desc='Reading in Validation Data'):
        image_folder = os.path.join(VALIDATION_PATH, letter)
        for im_file in os.listdir(image_folder):
            image = PIL.Image.open(os.path.join(image_folder, im_file))
            image_arr = np.array(image)
            # Get WINDOW_SIZE wide window and normalize
            padding = (image_arr.shape[1] - WINDOW_SIZE) // 2
            image_arr = image_arr[:, padding:padding+WINDOW_SIZE] // 255
            assert image_arr.shape[1] == WINDOW_SIZE
            # Invert image to make blackground white
            image_arr = np.where(image_arr == 0, 1, 0)
            validation_ds.append(image_arr)
            validation_labels.append(one_hot_encoded_labels[letter])
        enumerator += 1

    return np.array(train_ds), np.array(train_labels), np.array(test_ds), np.array(test_labels), \
           np.array(validation_ds), np.array(validation_labels), np.array(letters)


def shuffle_data(train_ds, train_labels):
    """
    Shuffles the data and labels
    :param train_ds: training data
    :param train_labels: training labels
    :return: shuffled data and labels
    """
    assert len(train_ds) == len(train_labels)
    p = np.random.permutation(len(train_ds))
    return train_ds[p], train_labels[p]


# adapted from
# https://towardsdatascience.com/exploring-confusion-matrix-evolution-on-tensorboard-e66b39f4ac12
def plot_confusion_matrix(cm, class_names):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.

    Args:
       cm (array, shape = [n, n]): a confusion matrix of integer classes
       class_names (array, shape = [n]): String names of the integer classes
    """

    figure = plt.figure(figsize=(18, 18))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Normalize the confusion matrix.
    cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


if __name__ == "__main__":
    print("[INFO] Reading in Data")
    train_data, train_labels, test_data, test_labels, validation_data, validation_labels, class_labels = read_in_data()
    print("[INFO] Shuffling Data")
    train_data, train_labels = shuffle_data(train_data, train_labels)
    test_data, test_labels = shuffle_data(test_data, test_labels)
    validation_data, validation_labels = shuffle_data(validation_data, validation_labels)

    # define parameter-sweep
    architectures = [val for val in range(2)]
    dropout_rates = [0.2, 0.4, 0.6]
    last_dense_layer_sizes = [96, 128, 256]
    learning_rates = [0.001, 0.01, 0.1]

    # param sweep result aggregation
    best_model = (None, 0, '')
    column_headings = ['Architecture', 'Dropout Rate', 'Last Dense Layer Size', 'Learning Rate',
                       'Accuracy', 'Precision', 'Recall', 'F Score', 'Support']
    run_details = []

    for architecture, dropout_rate, dense_size, learning_rate \
            in itertools.product(architectures, dropout_rates, last_dense_layer_sizes, learning_rates):

        best_model_i = (None, 0, -1)

        for i in range(num_models):
            print(f"[INFO] Constructing Model {i}")

            model = get_model(
                arch=architecture,
                last_layer_size=dense_size,
                verbose=i == 0,
                dropout_rate=dropout_rate)
            # model = get_model(verbose=i == 0)

            compile_model(model, learning_rate)

            print("[INFO] Beginning Model Training")
            early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1)
            # tb = TensorBoard(
            #     log_dir=f'models/train_model{i}_logs',
            #     histogram_freq=1,
            #     write_graph=True,
            #     write_images=False,
            #     write_steps_per_second=False,
            #     update_freq='epoch',
            #     profile_batch=0,
            #     embeddings_freq=0,
            #     embeddings_metadata=None
            # )
            cbks = [early_stopping]
            _ = model.fit(train_data, train_labels, epochs=epochs, callbacks=cbks,
                          batch_size=batch_size,
                          validation_data=(validation_data, validation_labels))
            try:
                model.save(f'models/trained_model{i}')
            except:
                print("[ERROR] Could not save model")

            print("[INFO] Generating Predictions")
            test_pred_raw = model.predict(test_data)

            test_pred = np.argmax(test_pred_raw, axis=1)
            rounded_labels = np.argmax(test_labels, axis=1)
            accuracy = accuracy_score(rounded_labels, test_pred)
            print(f"[RESULT] Accuracy: {accuracy}")
            precision, recall, fscore, support = precision_recall_fscore_support(rounded_labels, test_pred)
            print(f"[RESULT] Precision: {precision}")
            print(f"[RESULT] Recall: {recall}")
            print(f"[RESULT] F-Score: {fscore}")
            print(f"[RESULT] Support: {support}")
            # Calculate the confusion matrix using sklearn.metrics
            cm = confusion_matrix(rounded_labels, test_pred)
            plot_confusion_matrix(cm, class_labels)

            run_details.append([architecture, dropout_rate, dense_size, learning_rate,
                                accuracy, precision, recall, fscore, support])

            # Only save model if it performs better
            if accuracy > best_model_i[1]:
                print(f"[INFO] Model i ({i}) performing better than previous best model i ({best_model_i[2]})")
                best_model_i = (model, accuracy, i)

        if best_model_i[1] > best_model[1]:
            print(
                f"[INFO] Best model i ({best_model_i[2]}) performing better than previous best model ({best_model[2]})")
            name = f'{architecture}_{dropout_rate*100}_{dense_size}_{learning_rate*1000}_{best_model_i[2]}'
            best_model_i[0].save(f'models/best_model_{name}')
            best_model = (best_model_i[0], best_model_i[1], name)

    print(f"[INFO] Saving Best Model {best_model[2]}")
    best_model[0].save(f'models/best_model_sweep')

    csv = CSVWriter(filename='iam_results',
                    column_names=column_headings,
                    data_values=run_details
                    )
    csv.create_csv_file()
