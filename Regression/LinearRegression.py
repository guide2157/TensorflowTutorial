import numpy as np
import tensorflow.compat.v1 as tf
from sklearn import linear_model
import utils


class LinearRegression():
    def __init__(self, session, loss="l2", lammy=1, num_epochs=40, learning_rate=0.1, verbose=0):
        self.loss = loss
        self.lammy = lammy
        self.num_epochs = num_epochs
        self.verbose = verbose
        self.learning_rate = learning_rate
        self.w = None
        self.b = None
        self.session = session
        tf.disable_eager_execution()

    def fit(self, X, y):
        x_tf = tf.placeholder(tf.float32, shape=X.shape)
        y_tf = tf.placeholder(tf.float32, shape=y.shape)
        self.w = tf.get_variable("W", (X.shape[1], 1), tf.float32, tf.zeros_initializer())
        self.b = tf.get_variable("B", (1, 1), tf.float32, tf.zeros_initializer())
        cost = self.compute_cost(x_tf, y_tf)
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(cost)
        init = tf.global_variables_initializer()
        self.session.run(init)
        last_cost = 0
        for epoch in range(self.num_epochs + 1):
            _, epoch_cost = self.session.run([optimizer, cost], feed_dict={x_tf: X, y_tf: y})
            if abs(epoch_cost - last_cost) < 1e-10:
                break
            last_cost = epoch_cost
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
            return rsq + self.lammy * tf.reduce_sum(self.w ** 2) / 2
        else:
            return rsq + self.lammy * tf.reduce_sum(tf.abs(self.w))


if __name__ == '__main__':
    X_train, X_test, y_train, y_test = utils.preprocess_avocado()

    with tf.Session() as sess:
        model = LinearRegression(sess, verbose=1, loss='l1', learning_rate=0.01, num_epochs=500)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)

    difference = np.mean(np.square(pred - np.array(y_test)))
    print("The test error is: ", difference)

    model = linear_model.LinearRegression()
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    difference = np.mean(np.square(pred - np.array(y_test)))
    print("The test error from sk-learn is: ", difference)
