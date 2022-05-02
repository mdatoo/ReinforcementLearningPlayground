from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten


class Linear:
    @staticmethod
    def build(hidden: int, classes: int) -> Sequential:
        model = Sequential()

        model.add(Flatten())
        model.add(Dense(hidden, activation="relu"))
        model.add(Dense(classes, activation="softmax"))

        return model
