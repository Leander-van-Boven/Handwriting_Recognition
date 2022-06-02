from tensorflow.python.keras import models, layers, losses
from tensorflow.python.keras.layers import PReLU, Softmax


# 5 epochs accuracy 91%
# 10 epochs same
# new data only at 90%
def get_model(num_classes=28, input_shape=(71, 71, 3)):
    model = models.Sequential()

    model.add(layers.Conv2D(filters=96, kernel_size=(3, 3), activation=PReLU(), input_shape=input_shape))
    model.add(layers.MaxPooling2D((3, 3), strides=2))
    model.add(layers.Conv2D(filters=128, kernel_size=(3, 3), activation=PReLU()))
    model.add(layers.MaxPooling2D((3, 3), strides=2))
    model.add(layers.Conv2D(filters=160, kernel_size=(3, 3), activation=PReLU()))
    model.add(layers.MaxPooling2D((3, 3), strides=2))
    model.add(layers.Conv2D(filters=256, kernel_size=(3, 3), activation=PReLU()))
    model.add(layers.Conv2D(filters=256, kernel_size=(3, 3), activation=PReLU()))
    model.add(layers.Flatten())
    model.add(layers.Dense(1024, activation=PReLU()))
    # output layer
    output_layer = layers.Dense(num_classes, activation=Softmax())
    model.add(output_layer)
    # model.summary()
    return model


def compile_model(model):
    model.compile(
        optimizer='adam',
        loss=losses.CategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
