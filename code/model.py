import tensorflow as tf
import numpy as np


class TumorClassifier(tf.keras.Model):

    def __init__(self):
        super().__init__()

        self.batch_size = 32
        self.num_classes = 4
        self.loss_list = []
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

        # layers of model
        self.conv1 = tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=3,
            strides=1,
            activation='relu')
        self.max_pool1 = tf.keras.layers.MaxPool2D()
        self.dropout1 = tf.keras.layers.Dropout(0.25)
        self.conv2 = tf.keras.layers.Conv2D(
            filters=64,
            kernel_size=3,
            strides=1,
            activation='relu')
        self.max_pool2 = tf.keras.layers.MaxPool2D()
        self.dropout2 = tf.keras.layers.Dropout(0.25)
        self.conv3 = tf.keras.layers.Conv2D(
            filters=128,
            kernel_size=3,
            strides=1,
            activation='relu')
        self.dropout3 = tf.keras.layers.Dropout(0.4)
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(128)
        self.dropout4 = tf.keras.layers.Dropout(0.3)
        self.dense2 = tf.keras.layers.Dense(4)
        self.softmax = tf.keras.layers.Softmax()

    def call(self, inputs, is_training=True):
        x = tf.cast(inputs, dtype=tf.float32)
        x = self.conv1(x)
        x = self.max_pool1(x)
        if is_training:
            x = self.dropout1(x)
        x = self.conv2(x)
        x = self.max_pool2(x)
        if is_training:
            x = self.dropout2(x)
        x = self.conv3(x)
        if is_training:
            x = self.dropout3(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dropout4(x)
        x = self.dense2(x)
        probs = self.softmax(x)
        return probs
    
    def accuracy(self, logits, y_true):
        # y_pred = tf.argmax(probs, axis=1, output_type=tf.int64)
        # y_true = tf.argmax(y_true, axis=1)
        # print("y_pred shape", y_pred.shape)
        # print("y_true shape", y_true.shape)

        # return tf.metrics.Accuracy()(y_true, y_pred)
        correct_predictions = tf.equal(tf.argmax(logits, 1), tf.argmax(y_true, 1))
        return tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

