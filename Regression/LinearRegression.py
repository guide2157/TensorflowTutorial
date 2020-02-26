import pandas as pd
import numpy as np
import matplotlib as plt
import tensorflow.compat.v1 as tf
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


class LinearRegression():
    def __init__(self, loss="l2", lammy=1, num_epochs=40, learning_rate=0.1, verbose=0):
        self.loss = loss
        self.lammy = lammy
        self.num_epochs = num_epochs
        self.verbose = verbose
        self.learning_rate = learning_rate
        self.w = None
        self.b = None
        self.session = None
        tf.disable_eager_execution()

    def fit(self, X, y):
        x_tf = tf.placeholder(tf.float32, shape=X.shape)
        y_tf = tf.placeholder(tf.float32, shape=y.shape)
        self.w = tf.get_variable("W", (X.shape[1], 1), tf.float32, tf.zeros_initializer())
        self.b = tf.get_variable("B", (1, 1), tf.float32, tf.zeros_initializer())
        cost = self.compute_cost(x_tf, y_tf)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(cost)
        init = tf.global_variables_initializer()
        self.session = tf.Session()
        self.session.run(init)
        for epoch in range(self.num_epochs):
            _, epoch_cost = self.session.run([optimizer, cost], feed_dict={x_tf: X, y_tf: y})
            if self.verbose > 0:
                print("Cost after epoch %i: %f" % (epoch, epoch_cost))

    def predict(self, X):
        x = tf.placeholder(tf.float32, shape=X.shape)
        prediction = tf.add(tf.matmul(x, self.w), self.b)
        result = self.session.run(prediction, feed_dict={x: X})
        return result

    def compute_cost(self, X, y):
        result = tf.add(tf.matmul(X, self.w), self.b)
        rsq = tf.reduce_mean((result - y) ** 2)
        if self.loss == 'l2':
            return rsq + self.lammy * tf.reduce_sum(self.w**2) / 2
        else:
            return rsq + self.lammy * tf.reduce_sum(tf.abs(self.w))


if __name__ == '__main__':
    dataset = pd.read_csv("avocado.csv")
    dataset.drop(['Date', 'Total Bags', 'Small Bags', 'Large Bags', 'XLarge Bags', 'year', 'region'],
                 inplace=True, axis=1)
    dataset.drop(dataset.columns[[0]], axis=1, inplace=True)

    le = LabelEncoder()
    dataset['type'] = le.fit_transform(dataset['type'])
    onehot = OneHotEncoder()
    type = dataset['type'].values
    type_encoded = onehot.fit_transform(dataset.type.values.reshape(-1, 1)).toarray()
    dataset.drop(['type'], inplace=True, axis=1)
    encoded_data = pd.DataFrame(np.concatenate([dataset, type_encoded], axis=1))

    y = encoded_data[0]
    X = encoded_data.drop([0], axis=1)

    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(X))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression(verbose=1, loss='l1', learning_rate=0.1, num_epochs=60)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    difference = np.mean(np.square(pred - np.array(y_test)))
    print("The test error is: ", difference)
