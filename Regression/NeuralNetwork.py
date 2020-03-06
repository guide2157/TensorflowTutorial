import tensorflow.compat.v1 as tf
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.linear_model import LinearRegression
import utils


class NNRegression():
    def __init__(self, num_layers=3, layer_size=64, activation="relu", epoch=200):
        self.num_layers = num_layers
        self.activation = activation
        self.layer_size = layer_size
        self.epoch = epoch
        self.model = None

    def create_model(self, X):
        self.model = Sequential()
        self.model.add(Dense(self.layer_size, input_shape=[X.shape[1]]))
        for i in range(self.num_layers - 1):
            self.model.add(Dense(self.layer_size, activation=self.activation))
        self.model.add(Dense(1))

        optimizer = tf.keras.optimizers.RMSprop(0.001)
        self.model.compile(loss='mse',
                           optimizer=optimizer,
                           metrics=['mae', 'mse'])

    def fit(self, X, y):
        self.create_model(X)
        self.model.fit(X, y,
                  epochs=self.epoch, validation_split=0.2, verbose=0)

    def predict(self, X):
        return self.model.predict(X)


if __name__ == '__main__':
    X_train, X_test, y_train, y_test = utils.preprocess_avocado()

    model = NNRegression()
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    difference = np.mean(np.square(pred - np.array(y_test)))
    print("The test error is: ", difference)

    model = LinearRegression()
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    difference = np.mean(np.square(pred - np.array(y_test)))
    print("The test error from sk-learn is: ", difference)
