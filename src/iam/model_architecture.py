import string

from tensorflow.python.keras import models, layers, losses
from tensorflow.python.keras.layers import PReLU, Softmax

import torch.nn as nn
import torchvision.models as models



def get_model(num_classes=62, input_shape=(128, 128, 3)):
    # TODO: add model architecture here
    # TODO: input shape to be determined @jesper
    # The input shape of the training images of NIST are 128x128
    # TODO: should have len(string.ascii_letters + string.digits) classes = 62
    model = models.Sequential()

    model.add(layers.Conv2D(filters=128, kernel_size=(3, 3), activation=PReLU(), input_shape=input_shape))
    model.add(layers.MaxPooling2D((3, 3), strides=2))
    model.add(layers.Conv2D(filters=128, kernel_size=(3, 3), activation=PReLU()))
    model.add(layers.MaxPooling2D((3, 3), strides=2))
    model.add(layers.Conv2D(filters=128, kernel_size=(3, 3), activation=PReLU()))
    model.add(layers.MaxPooling2D((3, 3), strides=2))
    model.add(layers.Conv2D(filters=256, kernel_size=(3, 3), activation=PReLU()))
    #model.add(layers.MaxPooling2D((3, 3), strides=2))
    model.add(layers.Conv2D(filters=256, kernel_size=(3, 3), activation=PReLU()))
    # model.add(layers.MaxPooling2D((3, 3), strides=2))
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation=PReLU()))

    # output layer
    output_layer = layers.Dense(num_classes, activation=PReLU())
    model.add(output_layer)

    return model
######################################################################################

def build_model(fine_tune=True,num_classes=62, input_shape=(128, 128, 3)):
    model = models.efficientnet_b0(pretrained=True)

    if fine_tune:
        for params in model.parameters():
            params.requires_grad = True
    else:
        for params in model.parameters():
            params.requires_grad = False

    model.classifier[1] = nn.Linear(in_features=1280, out_features=num_classes)

    return model
######################################################################################

def compile_model(model):
    model.compile(
        optimizer='adam',
        loss=losses.CategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
