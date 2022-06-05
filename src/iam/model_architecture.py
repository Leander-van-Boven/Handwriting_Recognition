from tensorflow.python.keras import Model, Sequential
from tensorflow.keras.layers import Dense
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Flatten, PReLU, Dropout
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


# def get_model(num_classes=63, input_shape=(128, 64, 3), transfer_learning=True, verbose=False):
#     base_model = EfficientNetV2B3(
#         include_top=False,
#         weights='imagenet',
#         input_tensor=None,
#         input_shape=input_shape,
#         pooling='max'
#     )
#     # base_model.trainable = not transfer_learning
#     base_model.trainable = True
#     output_layer = Dense(num_classes, activation='softmax')(base_model.output)
#     model = Model(inputs=base_model.input, outputs=output_layer)
#
#     if verbose:
#         print(model.summary())
#
#     return model


def get_model(num_classes=63,
              input_shape=(128, 64, 3),
              arch:int = 0,
              big_model:bool = False,
              dropout_rate:float = 0.4,
              last_layer_size:int = 96,
              activation_function:callable = PReLU,
              verbose=False):
    model = Sequential()

    if arch == 0:
        model.add(Conv2D(filters=32, kernel_size=3, input_shape=input_shape))
        model.add(activation_function())
        model.add(Dropout(dropout_rate))
        model.add(MaxPooling2D(2, 2))
        model.add(Conv2D(filters=64, kernel_size=3))
        model.add(activation_function())
        model.add(Dropout(dropout_rate))
        model.add(MaxPooling2D(2, 2))
        model.add(Flatten())
        model.add(Dense(last_layer_size))
        model.add(activation_function())
        model.add(Dropout(dropout_rate))
        model.add(Dense(num_classes, activation='softmax'))

    elif arch == 1:
        # create different architecture
        pass

    # create more archs...

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
