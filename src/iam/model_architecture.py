import string

from tensorflow.python.keras import models, layers, losses
from tensorflow.python.keras.layers import PReLU, Softmax


def get_model(num_classes=62, input_shape=(128, 128, 3)):
    # TODO: add model architecture here
    # TODO: input shape to be determined @jesper
    # The input shape of the training images of NIST are 128x128
    # TODO: should have len(string.ascii_letters + string.digits) classes = 62
    model = models.Sequential()

    model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), activation=PReLU(), input_shape=input_shape))
    model.add(layers.MaxPooling2D((3, 3), strides=2))
    model.add(layers.Conv2D(filters=128, kernel_size=(3, 3), activation=PReLU()))
    model.add(layers.MaxPooling2D((3, 3), strides=2))
    model.add(layers.Conv2D(filters=256, kernel_size=(3, 3), activation=PReLU()))
    model.add(layers.MaxPooling2D((3, 3), strides=2))
    model.add(layers.Conv2D(filters=512, kernel_size=(3, 3), activation=PReLU()))
    model.add(layers.MaxPooling2D((3, 3), strides=2))
    model.add(layers.Conv2D(filters=1024, kernel_size=(3, 3), activation=PReLU()))
    model.add(layers.MaxPooling2D((3, 3), strides=2))
    model.add(layers.Flatten())
    model.add(layers.Dense(1024, activation=PReLU))

    # output layer
    output_layer = layers.Dense(num_classes, activation=Softmax())
    model.add(output_layer)
    return None

def compile_model(model):
    model.compile(
        optimizer='adam',
        loss=losses.CategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )