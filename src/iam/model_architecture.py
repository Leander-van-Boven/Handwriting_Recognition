from tensorflow.python.keras import models, layers, losses, Model
from tensorflow.keras.layers import Dense
from tensorflow.python.keras.losses import CategoricalCrossentropy
from keras.applications.efficientnet_v2 import EfficientNetV2B3


# def get_model(num_classes=62, input_shape=(128, 128, 3)):
#     # TODO: add model architecture here
#     # TODO: input shape to be determined @jesper
#     # The input shape of the training images of NIST are 128x128
#     # TODO: should have len(string.ascii_letters + string.digits) classes = 62
#     model = models.Sequential()
#
#     model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), activation=PReLU(), input_shape=input_shape))
#     model.add(layers.MaxPooling2D((3, 3), strides=2))
#     model.add(layers.Conv2D(filters=128, kernel_size=(3, 3), activation=PReLU()))
#     model.add(layers.MaxPooling2D((3, 3), strides=2))
#     model.add(layers.Conv2D(filters=256, kernel_size=(3, 3), activation=PReLU()))
#     model.add(layers.MaxPooling2D((3, 3), strides=2))
#     model.add(layers.Conv2D(filters=512, kernel_size=(3, 3), activation=PReLU()))
#     model.add(layers.MaxPooling2D((3, 3), strides=2))
#     model.add(layers.Conv2D(filters=1024, kernel_size=(3, 3), activation=PReLU()))
#     model.add(layers.MaxPooling2D((3, 3), strides=2))
#     model.add(layers.Flatten())
#     model.add(layers.Dense(1024, activation=PReLU))
#
#     # output layer
#     output_layer = layers.Dense(num_classes, activation=Softmax())
#     model.add(output_layer)
#     return None


def get_model(num_classes=63, input_shape=(128, 100, 1), transfer_learning=True, verbose=False):
    base_model = EfficientNetV2B3(
        include_top=False,
        weights='imagenet',
        input_tensor=None,
        input_shape=input_shape,
        pooling='max'
    )
    base_model.trainable = not transfer_learning
    output_layer = Dense(num_classes, activation='softmax')(base_model.output)
    model = Model(inputs=base_model.input, outputs=output_layer)

    if verbose:
        print(model.summary())

    return model


def compile_model(model):
    model.compile(
        optimizer='adam',
        loss=CategoricalCrossentropy(),  # from_logits=True, disabled because of Softmax
        metrics=['accuracy']
    )


if __name__ == '__main__':
    model = get_model(verbose=True)
    # compile_model(model)
