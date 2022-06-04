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
from src.dss import get_model
from src.dss.model import compile_model, read_in_data

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


if __name__ == "__main__":
    print("[INFO] Reading in Data")
    train_data, train_labels, test_data, test_labels, validation_data, validation_labels, class_lables = read_in_data()

    print("[INFO] Constructing Model")
    model = get_model()
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