import tensorflow as tf
import numpy as np


class TumorClassifier(tf.keras.Model):

    def __init__(self):
        super(TumorClassifier, self).__init__()

        self.num_classes = 4
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
        # self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

        # layers of model
        self.conv1 = tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=3,
            strides=1)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.relu1 = tf.keras.layers.LeakyReLU()
        self.max_pool1 = tf.keras.layers.MaxPool2D()
        self.dropout1 = tf.keras.layers.Dropout(0.4)
        self.conv2 = tf.keras.layers.Conv2D(
            filters=64,
            kernel_size=3,
            strides=1)
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.relu2 = tf.keras.layers.LeakyReLU()
        self.max_pool2 = tf.keras.layers.MaxPool2D()
        self.dropout2 = tf.keras.layers.Dropout(0.5)
        self.conv3 = tf.keras.layers.Conv2D(
            filters=128,
            kernel_size=3,
            strides=1)
        self.bn3 = tf.keras.layers.BatchNormalization()
        self.relu3 = tf.keras.layers.LeakyReLU()
        self.dropout3 = tf.keras.layers.Dropout(0.4)
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(128)
        self.dropout4 = tf.keras.layers.Dropout(0.3)
        self.dense2 = tf.keras.layers.Dense(self.num_classes)
        self.softmax = tf.keras.layers.Softmax()

    def call(self, inputs, is_training=True):
        x = self.conv1(inputs)
        if is_training:
            x = self.bn1(x)
        x = self.relu1(x)
        x = self.max_pool1(x)
        if is_training:
            x = self.dropout1(x)
        x = self.conv2(x)
        if is_training:
            x = self.bn2(x)
        x = self.relu2(x)
        x = self.max_pool2(x)
        if is_training:
            x = self.dropout2(x)
        x = self.conv3(x)
        if is_training:
            x = self.bn3(x)
        x = self.relu3(x)
        if is_training:
            x = self.dropout3(x)
        x = self.flatten(x)
        x = self.dense1(x)
        if is_training:
            x = self.dropout4(x)
        x = self.dense2(x)
        probs = self.softmax(x)
        return probs
    
    # def __init__(self):
    #     super().__init__()

    #     self.num_classes = 4

    #     # layers of model
    #     self.basic = tf.keras.models.Sequential()
    #     self.basic.add(tf.keras.layers.Conv2D(
    #         filters=32,
    #         kernel_size=3,
    #         strides=1))
    #     self.basic.add(tf.keras.layers.ReLU())
    #     self.basic.add(tf.keras.layers.MaxPool2D())
    #     # self.basic.add(tf.keras.layers.Dropout(0.25))
    #     self.basic.add(tf.keras.layers.Conv2D(
    #         filters=64,
    #         kernel_size=3,
    #         strides=1
    #     ))
    #     self.basic.add(tf.keras.layers.ReLU())
    #     self.basic.add(tf.keras.layers.MaxPool2D())
    #     # self.basic.add(tf.keras.layers.Dropout(0.25))
    #     self.basic.add(tf.keras.layers.Conv2D(
    #         filters=128,
    #         kernel_size=3,
    #         strides=1
    #     ))
    #     self.basic.add(tf.keras.layers.ReLU())
    #     self.basic.add(tf.keras.layers.Flatten())
    #     self.basic.add(tf.keras.layers.Dense(128))
    #     self.basic.add(tf.keras.layers.Dense(self.num_classes))
    #     self.basic.add(tf.keras.layers.Softmax())

    # def call(self, inputs, is_training=True):
    #     x = tf.cast(inputs, dtype=tf.float32)
    #     probs = self.basic(x)
    #     return probs

    def accuracy(self, logits, y_true):
        acc = tf.keras.metrics.SparseCategoricalAccuracy()
        return acc(y_true, logits)
        #correct_predictions = tf.math.equal(tf.argmax(logits, 1), y_true)
        #return tf.reduce_mean(tf.cast(correct_predictions, tf.float64))
    
    def loss(self, y_true, probs):
        losses = tf.keras.losses.SparseCategoricalCrossentropy()(y_true, probs)
        avg_loss = tf.math.reduce_mean(losses)
        return avg_loss