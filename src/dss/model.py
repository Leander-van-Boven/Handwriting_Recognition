import itertools
from pathlib import Path

from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support
from matplotlib import pyplot as plt
from tensorflow.python.keras import callbacks
import os
import PIL
from tqdm import tqdm
import numpy as np

from src.dss.model_architecture import get_model, compile_model
from src.utils.csv_writer import CSVWriter

batch_size = 16
epochs = 25
num_models = 5

parent_dir = Path('../../data/dss/FINAL_IMAGES_AUGMENTS').resolve()
TRAIN_PATH = os.path.join(parent_dir, 'Train')
TEST_PATH = os.path.join(parent_dir, 'Test')
VALIDATION_PATH = os.path.join(parent_dir, 'Validation')
num_images = (1014, 234, 234)


# label files are one hot encoded
def read_in_data():
    letters = []

    train_ds = []
    test_ds = []
    validation_ds = []

    train_labels = []
    test_labels = []
    validation_labels = []

    for letter in os.listdir(TRAIN_PATH):
        letters.append(letter)

    one_hot_encoded_labels = []
    for i in range(len(letters)):  # == list of 28
        tmp = []
        for j in range(len(letters)):
            if j == i:
                tmp.append(1)
            else:
                tmp.append(0)
        one_hot_encoded_labels.append(tmp)

    ENUMERATOR = 0
    for letter in tqdm(os.listdir(TRAIN_PATH), desc='Reading in Training Data'):
        # tmp = []
        image_folder = os.path.join(TRAIN_PATH, letter)
        for counter, im_file in enumerate(os.listdir(image_folder)):
            if counter >= num_images[0]:  # 26*39
                break
            # image = cv2.imread(os.path.join(image_folder, im_file))
            # tmp.append(image)
            image = PIL.Image.open(os.path.join(image_folder, im_file))
            image_arr = np.array(image)
            # tmp.append(image_arr)
            # Get 40 wide window and normalize
            image_arr = np.expand_dims(image_arr[:, 15:55, 0] // 255, axis=2)
            # Invert image to make black background white
            image_arr = np.where(image_arr == 0, 1, 0)
            train_ds.append(image_arr)
            train_labels.append(one_hot_encoded_labels[ENUMERATOR])
        # train_ds.append(tmp)
        ENUMERATOR += 1

    ENUMERATOR = 0
    for letter in tqdm(os.listdir(TEST_PATH), desc='Reading in Test Data'):
        image_folder = os.path.join(TEST_PATH, letter)
        tmp = []
        for counter, im_file in enumerate(os.listdir(image_folder)):
            if counter >= num_images[2]:  # 26*7
                break
            # image = cv2.imread(os.path.join(image_folder, im_file))
            # tmp.append(image)
            image = PIL.Image.open(os.path.join(image_folder, im_file))
            image_arr = np.array(image)
            # tmp.append(image_arr)
            # Get 40 wide window and normalize
            image_arr = np.expand_dims(image_arr[:, 15:55, 0] // 255, axis=2)
            # Invert image to make black background white
            image_arr = np.where(image_arr == 0, 1, 0)
            test_ds.append(image_arr)
            test_labels.append(one_hot_encoded_labels[ENUMERATOR])
        # test_ds.append(tmp)
        ENUMERATOR += 1

    ENUMERATOR = 0
    for letter in tqdm(os.listdir(VALIDATION_PATH), desc='Reading in Validation Data'):
        image_folder = os.path.join(VALIDATION_PATH, letter)
        tmp = []
        for counter, im_file in enumerate(os.listdir(image_folder)):
            if counter >= num_images[1]:  # 26*7
                break
            # image = cv2.imread(os.path.join(image_folder, im_file))

            # tmp.append(image)
            image = PIL.Image.open(os.path.join(image_folder, im_file))
            image_arr = np.array(image)
            # tmp.append(image_arr)
            # Get 40 wide window and normalize
            image_arr = np.expand_dims(image_arr[:, 15:55, 0] // 255, axis=2)
            # Invert image to make black background white
            image_arr = np.where(image_arr == 0, 1, 0)
            validation_ds.append(image_arr)
            validation_labels.append(one_hot_encoded_labels[ENUMERATOR])
        # validation_ds.append(tmp)
        ENUMERATOR += 1

    return np.array(train_ds), np.array(train_labels), np.array(test_ds), np.array(test_labels), \
           np.array(validation_ds), np.array(validation_labels), np.array(letters)


def shuffle_data(train_ds, train_labels):
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

    architectures = [val for val in range(1)]
    big_models = [False, True]
    dropout_rates = [0.2, 0.4, 0.6, 0.8]
    last_dense_layer_sizes = [64, 96, 128]

    column_headings = ['Architecture', 'Big Model', 'Dropout Rate', 'Last Dense Layer Size',
                       'Accuracy', 'Precision', 'Recall', 'F Score', 'Support']
    run_details = []

    for architecture, use_big_model, dropout_rate, dense_size \
            in itertools.product(architectures, big_models, dropout_rates, last_dense_layer_sizes):
        for i in range(num_models):
            print(f"[INFO] Constructing Model {i}")
            model = get_model(
                arch=architecture,
                big_model=use_big_model,
                last_layer_size=dense_size,
                verbose= i == 0)
            compile_model(model)

            print("[INFO] Beginning Model Training")
            early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=1)
            # tb = callbacks.TensorBoard(
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
            callbacks = [early_stopping]
            _ = model.fit(train_data, train_labels, epochs=epochs, callbacks=callbacks,
                          batch_size=batch_size,
                          validation_data=(validation_data, validation_labels))

            print("[INFO] Generating Predictions")
            test_pred_raw = model.predict(test_data)

            test_pred = np.argmax(test_pred_raw, axis=1)
            rounded_labels = np.argmax(test_labels, axis=1)
            accuracy = accuracy_score(rounded_labels, test_pred)
            print("[RESULT] Accuracy: %f" % accuracy)
            precision, recall, fscore, support = precision_recall_fscore_support(rounded_labels, test_pred)
            print(f"[RESULT] Precision: {precision}")
            print(f"[RESULT] Recall: {recall}")
            print(f"[RESULT] F-Score: {fscore}")
            print(f"[RESULT] Support: {support}")
            # Calculate the confusion matrix using sklearn.metrics
            cm = confusion_matrix(rounded_labels, test_pred)
            plot_confusion_matrix(cm, class_labels)

            run_details.append([architecture, use_big_model, dropout_rate, dense_size,
                               accuracy, precision, recall, fscore, support])

            # Only save model if it performs better
            model.save(f'models/trained_model{i}')

    csv = CSVWriter(filename='dss_results',
                    column_names=column_headings,
                    data_values=run_details
                    )
    csv.create_csv_file()
