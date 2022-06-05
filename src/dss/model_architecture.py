from tensorflow.python.keras.losses import CategoricalCrossentropy
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, PReLU, Softmax, Dropout
from keras.layers import BatchNormalization

# 5 epochs accuracy 91%
# 10 epochs same
# new data only at 90%
# def get_model(num_classes=28, input_shape=(71, 71, 3)):
#     model = Sequential()
#
#     model.add(Conv2D(filters=96, kernel_size=(3, 3), activation=PReLU(), input_shape=input_shape))
#     model.add(MaxPooling2D((3, 3), strides=2))
#     model.add(Conv2D(filters=128, kernel_size=(3, 3), activation=PReLU()))
#     model.add(MaxPooling2D((3, 3), strides=2))
#     model.add(Conv2D(filters=160, kernel_size=(3, 3), activation=PReLU()))
#     model.add(MaxPooling2D((3, 3), strides=2))
#     model.add(Conv2D(filters=256, kernel_size=(3, 3), activation=PReLU()))
#     model.add(Conv2D(filters=256, kernel_size=(3, 3), activation=PReLU()))
#     model.add(Flatten())
#     model.add(Dense(1024, activation=PReLU()))
#     # output layer
#     output_layer = Dense(num_classes, activation=Softmax())
#     model.add(output_layer)
#     # model.summary()
#     return model


def get_model(num_classes=28,
              input_shape=(71, 40, 1),
              arch:int = 0,
              big_model:bool = False,
              dropout_rate:float = 0.5,
              last_layer_size:int = 96,
              activation_function = PReLU(),
              verbose: bool = False):
    model = Sequential()

    if arch == 0:
        model.add(Conv2D(filters=32, kernel_size=2, activation=activation_function, input_shape=input_shape))
        model.add(BatchNormalization())
        if big_model:
            model.add(Conv2D(filters=32, kernel_size=3, activation=activation_function))
            model.add(BatchNormalization())
            model.add(Conv2D(filters=32, kernel_size=5, strides=2, padding='same', activation=activation_function))
            model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))

        model.add(Conv2D(filters=64, kernel_size=2, activation=activation_function))
        model.add(BatchNormalization())
        if big_model:
            model.add(Conv2D(filters=64, kernel_size=3, activation=activation_function))
            model.add(BatchNormalization())
            model.add(Conv2D(filters=64, kernel_size=5, strides=2, padding='same', activation=activation_function))
            model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))

        model.add(Conv2D(filters=128, kernel_size=3, activation=activation_function))
        model.add(BatchNormalization())
        model.add(Flatten())
        model.add(Dropout(dropout_rate))
        model.add(Dense(last_layer_size, activation=activation_function))
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
