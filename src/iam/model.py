from tensorflow.python.keras import models, layers, losses, callbacks
from tensorflow.python.keras.layers.advanced_activations import PReLU
import os
import PIL
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import torch

from src.iam.model_architecture import get_model, compile_model, get_model2, build_model
#from model_architecture import get_model, compile_model, get_model2, build_model

image_height = 128
image_width = 128
colour_channels = 3
input_shape = (image_height, image_width, colour_channels)
batch_size = 2
epochs = 15
num_classes = 62

# Assumes this file is in the same directory as the `parent_dir`
parent_dir = "IMData_Split"
TRAIN_PATH = os.path.join(parent_dir, 'Train')
TEST_PATH = os.path.join(parent_dir, 'Test')
VALIDATION_PATH = os.path.join(parent_dir, 'Validation')
output_layer = None

checkpoint_path = 'trained_model/trained_model.ckpt'
LEARNING_RATE = 0.001
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
    for i in range(len(letters)):  # == list of 62
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
            if counter >= 1014:
                break
            # image = cv2.imread(os.path.join(image_folder, im_file))
            # tmp.append(image)
            image = PIL.Image.open(os.path.join(image_folder, im_file))
            image_arr = np.array(image)
            # tmp.append(image_arr)
            train_ds.append(image_arr)
            train_labels.append(one_hot_encoded_labels[ENUMERATOR])
        # train_ds.append(tmp)
        ENUMERATOR += 1

    ENUMERATOR = 0
    for letter in tqdm(os.listdir(TEST_PATH), desc='Reading in Test Data'):
        image_folder = os.path.join(TEST_PATH, letter)
        tmp = []
        for im_file in os.listdir(image_folder):
            # image = cv2.imread(os.path.join(image_folder, im_file))
            # tmp.append(image)
            image = PIL.Image.open(os.path.join(image_folder, im_file))
            image_arr = np.array(image)
            # tmp.append(image_arr)
            test_ds.append(image_arr)
            test_labels.append(one_hot_encoded_labels[ENUMERATOR])
        # test_ds.append(tmp)
        ENUMERATOR += 1

    ENUMERATOR = 0
    for letter in tqdm(os.listdir(VALIDATION_PATH), desc='Reading in Validation Data'):
        image_folder = os.path.join(VALIDATION_PATH, letter)
        tmp = []
        for im_file in os.listdir(image_folder):
            # image = cv2.imread(os.path.join(image_folder, im_file))

            # tmp.append(image)
            image = PIL.Image.open(os.path.join(image_folder, im_file))
            image_arr = np.array(image)
            # tmp.append(image_arr)
            validation_ds.append(image_arr)
            validation_labels.append(one_hot_encoded_labels[ENUMERATOR])
        # validation_ds.append(tmp)
        ENUMERATOR += 1

    return np.array(train_ds), np.array(train_labels), np.array(test_ds), np.array(test_labels), np.array(
        validation_ds), np.array(validation_labels)
    # return train_ds, train_labels, test_ds, test_labels, validation_ds, validation_labels

######################################################################################
def test_model(model, test_data, test_labels):
    correct = 0
    total = 0

    for j in tqdm(range(len(test_data)), 'Testing The Model'):
        predicted = model.predict(test_data[j].reshape((1, 128, 128, 3)))[0]
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

    print("[RESULTS] Percentage Correct = %f" % (correct / total))


def train_model(model, train_loader, optimizer, criterion):
    model.train()
    train_running_loss = 0.0
    train_running_correct = 0
    counter = 0
    for i, data in tqdm(enumerate(train_loader), total=len(train_loader)):
        counter += 1
        image, labels = data
        image = image.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        # Forward pass.
        outputs = model(image)
        # Calculate the loss.
        loss = criterion(outputs, labels)
        train_running_loss += loss.item()
        # Calculate the accuracy.
        _, preds = torch.max(outputs.data, 1)
        train_running_correct += (preds == labels).sum().item()
        # Backpropagation
        loss.backward()
        # Update the weights.
        optimizer.step()

    # Loss and accuracy for the complete epoch.
    epoch_loss = train_running_loss / counter
    epoch_acc = 100. * (train_running_correct / len(train_loader.dataset))

    return epoch_loss, epoch_acc


def validate(model, validation_loader, criterion):
    model.eval()
    print('Validation')
    valid_running_loss = 0.0
    valid_running_correct = 0
    counter = 0
    with torch.no_grad():
        for i, data in tqdm(enumerate(validation_loader), total=len(validation_loader)):
            counter += 1

            image, labels = data
            image = image.to(device)
            labels = labels.to(device)
            # Forward pass.
            outputs = model(image)
            # Calculate the loss.
            loss = criterion(outputs, labels)
            valid_running_loss += loss.item()
            # Calculate the accuracy.
            _, preds = torch.max(outputs.data, 1)
            valid_running_correct += (preds == labels).sum().item()

    # Loss and accuracy for the complete epoch.
    epoch_loss = valid_running_loss / counter
    epoch_acc = 100. * (valid_running_correct / len(validation_loader.dataset))
    return epoch_loss, epoch_acc


def training(model, training_loader, validation_loader):
    device = ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Computation device: {device}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Epochs to train for: {EPOCHS}\n")

    model.to(device)

    # Optimizer.
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # Loss function.
    criterion = nn.CrossEntropyLoss()
    # Lists to keep track of losses and accuracies.
    train_loss, valid_loss = [], []
    train_acc, valid_acc = [], []
    # Start the training.

    for epoch in range(EPOCHS):
        print(f"[INFO]: Epoch {epoch + 1} of {epochs}")
        train_epoch_loss, train_epoch_acc = train_model(model, training_loader,
                                                  optimizer, criterion)
        valid_epoch_loss, valid_epoch_acc = validate(model, validation_loader,
                                                     criterion)

        train_loss.append(train_epoch_loss)
        valid_loss.append(valid_epoch_loss)
        train_acc.append(train_epoch_acc)
        valid_acc.append(valid_epoch_acc)
        print(f"Training loss: {train_epoch_loss:.3f}, training acc: {train_epoch_acc:.3f}")
        print(f"Validation loss: {valid_epoch_loss:.3f}, validation acc: {valid_epoch_acc:.3f}")
        print('-' * 50)


def _create_data_loader(training_x, training_y, validation_x, validation_y):
    train_loader = DataLoader(
        (training_x, training_y),
        batch_size = batch_size,
        shuffle=True
    )
    validation_loader = DataLoader(
        (validation_x, validation_y),
        batch_size=batch_size,
        shuffle=True
    )

    return train_loader, validation_loader

######################################################################################

if __name__ == "__main__":
    print("[INFO] Reading in Data")
    train_data, train_labels, test_data, test_labels, validation_data, validation_labels = read_in_data()

    print("[INFO] Creating Data Loaders")
    train_loader, validation_loader = _create_data_loader(train_data, train_labels, validation_data, validation_labels)

    print("[INFO] Constructing Model")
    #model = get_model()
    # compile_model(model)

    ######################################################################################
    model = build_model(fine_tune=True)

    training(model, train_loader, validation_loader)
    exit()s
    ######################################################################################

    print("[INFO] Beginning Model Training")
    cp_callback = callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                            save_weights_only=True,
                                            verbose=1)

    _ = model.fit(train_data, train_labels, epochs=EPOCHS,
                  validation_data=(validation_data, validation_labels),
                  callbacks=cp_callback,
                  batch_size=batch_size)

    print("[INFO] Generating Predictions")

    test_model(model, test_data, test_labels)
