from tensorflow.python.keras import models, layers, losses, callbacks
from tensorflow.python.keras.layers.advanced_activations import PReLU
import os
import PIL
from tqdm import tqdm
import numpy as np

from src.dss.model_architecture import get_model, compile_model

image_height = 71
image_width = 71
colour_channels = 3
input_shape = (image_height, image_width, colour_channels)
batch_size = 16
epochs = 15
num_classes = 27

# Assumes this file is in the same directory as the `parent_dir`
parent_dir = 'FINAL_IMAGES2'
TRAIN_PATH = os.path.join(parent_dir, 'Train')
TEST_PATH = os.path.join(parent_dir, 'Test')
VALIDATION_PATH = os.path.join(parent_dir, 'Validation')
output_layer = None

checkpoint_path = 'trained_model/trained_model.ckpt'
EPOCHS = 8


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
        for counter, im_file in enumerate(os.listdir(image_folder)):
            if counter >= 1014:
                break
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



    return np.array(train_ds), np.array(train_labels), np.array(test_ds), np.array(test_labels), np.array(validation_ds), np.array(validation_labels)
    # return train_ds, train_labels, test_ds, test_labels, validation_ds, validation_labels


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


if __name__ == "__main__":
    print("[INFO] Reading in Data")
    train_data, train_labels, test_data, test_labels, validation_data, validation_labels = read_in_data()

    print("[INFO] Constructing Model")
    model = get_model()
    compile_model(model)

    print("[INFO] Beginning Model Training")
    cp_callback = callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                            save_weights_only=True,
                                            verbose=1)
    _ = model.fit(train_data, train_labels, epochs=EPOCHS, callbacks=cp_callback)

    print("[INFO] Generating Predictions")

    test_model(model, test_data, test_labels)
