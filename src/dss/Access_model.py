import itertools

import sklearn.metrics
from tensorflow.python.keras import models, layers, losses, callbacks
from tensorflow.python.keras.layers.advanced_activations import PReLU
import os
import PIL
from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt

# accesses checkpoint of saved model and makes a confusion matrix

image_height = 71
image_width = 71
colour_channels = 3
input_shape = (image_height, image_width, colour_channels)
batch_size = 16
num_classes = 27

# Assumes this file is in the same directory as the `parent_dir`
parent_dir = 'FINAL_IMAGES_AUGMENTS'
TRAIN_PATH = os.path.join(parent_dir, 'Train')
TEST_PATH = os.path.join(parent_dir, 'Test')
VALIDATION_PATH = os.path.join(parent_dir, 'Validation')
output_layer = None

checkpoint_path = 'trained_model/trained_model.ckpt'
EPOCHS = 5


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

    class_lables = letters

    one_hot_encoded_labels = []
    for i in range(len(letters)):  # == list of 27
        tmp = []
        for j in range(len(letters)):
            if j == i:
                tmp.append(1)
            else:
                tmp.append(0)
        one_hot_encoded_labels.append(tmp)

    ENUMERATOR = 0
    for letter in tqdm(os.listdir(TRAIN_PATH), desc='Reading in Training Data'):
        #tmp = []
        image_folder = os.path.join(TRAIN_PATH, letter)
        for im_file in os.listdir(image_folder):
            #image = cv2.imread(os.path.join(image_folder, im_file))
            #tmp.append(image)
            image = PIL.Image.open(os.path.join(image_folder, im_file))
            image_arr = np.array(image)
            #tmp.append(image_arr)
            train_ds.append(image_arr)
            train_labels.append(one_hot_encoded_labels[ENUMERATOR])
        #train_ds.append(tmp)
        ENUMERATOR += 1

    ENUMERATOR = 0
    for letter in tqdm(os.listdir(TEST_PATH), desc='Reading in Test Data'):
        image_folder = os.path.join(TEST_PATH, letter)
        tmp = []
        for im_file in os.listdir(image_folder):
            #image = cv2.imread(os.path.join(image_folder, im_file))
            #tmp.append(image)
            image = PIL.Image.open(os.path.join(image_folder, im_file))
            image_arr = np.array(image)
            #tmp.append(image_arr)
            test_ds.append(image_arr)
            test_labels.append(one_hot_encoded_labels[ENUMERATOR])
        # test_ds.append(tmp)
        ENUMERATOR += 1

    ENUMERATOR = 0
    for letter in tqdm(os.listdir(VALIDATION_PATH), desc='Reading in Validation Data'):
        image_folder = os.path.join(VALIDATION_PATH, letter)
        tmp = []
        for im_file in os.listdir(image_folder):
            #image = cv2.imread(os.path.join(image_folder, im_file))

            # tmp.append(image)
            image = PIL.Image.open(os.path.join(image_folder, im_file))
            image_arr = np.array(image)
            #tmp.append(image_arr)
            validation_ds.append(image_arr)
            validation_labels.append(one_hot_encoded_labels[ENUMERATOR])
        #validation_ds.append(tmp)
        ENUMERATOR += 1



    return np.array(train_ds), np.array(train_labels), np.array(test_ds), np.array(test_labels), np.array(validation_ds), np.array(validation_labels), class_lables
    # return train_ds, train_labels, test_ds, test_labels, validation_ds, validation_labels


# 5 epochs accuracy 91%
# 10 epochs same
# new data only at 90%
def build_model():
    model = models.Sequential()

    model.add(layers.Conv2D(filters=96, kernel_size=(3, 3), activation=PReLU(), input_shape=input_shape))
    model.add(layers.MaxPooling2D((3, 3), strides=2))
    model.add(layers.Conv2D(filters=128, kernel_size=(3, 3), activation=PReLU()))
    model.add(layers.MaxPooling2D((3, 3), strides=2))
    model.add(layers.Conv2D(filters=160, kernel_size=(3, 3), activation=PReLU()))
    model.add(layers.MaxPooling2D((3, 3), strides=2))
    model.add(layers.Conv2D(filters=256, kernel_size=(3, 3), activation=PReLU()))
    model.add(layers.Conv2D(filters=256, kernel_size=(3, 3), activation=PReLU()))
    # model.add(layers.MaxPooling2D((3, 3), strides=1))
    # model.add(layers.Conv2D(filters=384, kernel_size=(3, 3), activation=PReLU()))
    # model.add(layers.Conv2D(filters=384, kernel_size=(3, 3), activation=PReLU()))
    # model.add(layers.MaxPooling2D((3, 3), strides=2))
    model.add(layers.Flatten())
    model.add(layers.Dense(1024, activation=PReLU()))
    # output layer
    output_layer = layers.Dense(num_classes, activation=PReLU())
    model.add(output_layer)
    model.summary()
    return model


def compile_model(model):
    model.compile(
        optimizer='adam',
        loss=losses.CategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )


def test_model(model, test_data, test_labels):
    correct = 0
    total = 0

    for j in tqdm(range(len(test_data)), 'Testing The Model'):
        predicted = model.predict(test_data[j].reshape((1, 71, 71, 3)))[0]
        # Predicted is an array of confidence values
        max_val = max(predicted)
        for i in range(len(predicted)):
            if predicted[i] == max_val:
                predicted[i] = 1
            else:
                predicted[i] = 0

        total += 1
        if (predicted == test_labels[j]).all():
            correct += 1

    print("[RESULTS] Percentage Correct = %f" % (correct/total))

def plot_model(history):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Validation'], loc='upper left')
    plt.savefig('accuracy_eff.png')
    plt.show()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Validation'], loc='upper left')
    plt.savefig('loss_eff.png')
    plt.show()


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
    train_data, train_labels, test_data, test_labels, validation_data, validation_labels, class_lables = read_in_data()

    print("[INFO] Constructing Model")
    model = build_model()
    compile_model(model)
    model.load_weights(checkpoint_path)

    print("[INFO] Generating Predictions")
    test_pred_raw = model.predict(test_data)

    test_pred = np.argmax(test_pred_raw, axis=1)
    rounded_labels = np.argmax(test_labels, axis=1)
    # Calculate the confusion matrix using sklearn.metrics
    cm = sklearn.metrics.confusion_matrix(rounded_labels, test_pred)
    print(cm)
    plot_confusion_matrix(cm, class_lables)