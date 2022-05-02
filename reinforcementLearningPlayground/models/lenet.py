from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D


class LeNet:
    @staticmethod
    def build(height: int, width: int, depth: int, classes: int) -> Sequential:
        model = Sequential()
        input_shape = (height, width, depth)

        model.add(Conv2D(20, (5, 5), padding="same", activation="relu", input_shape=input_shape))
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(20, (5, 5), padding="same", activation="relu"))
        model.add(MaxPooling2D((2, 2)))
        model.add(Flatten())
        model.add(Dense(500, activation="relu"))
        model.add(Dense(classes, activation="softmax"))

        return model
