import numpy as np
import tensorflow as tf
import pickle 
from pathlib import Path

def preprocess_data():

    path_test = Path("../data/Testing")
    path_train = Path("../data/Training")

    test_dataset = tf.keras.utils.image_dataset_from_directory(
        path_test, 
        color_mode="grayscale",
        batch_size=10000,
        shuffle=False)
    
    train_dataset = tf.keras.utils.image_dataset_from_directory(
        path_train, 
        color_mode="grayscale",
        batch_size=10000,
        shuffle=False)

    X_test = []
    Y_test = []
    X_train = []
    Y_train = []
    for image, label in test_dataset.take(1):
        X_test = image
        Y_test = label

    for image, label in train_dataset.take(1):
        X_train = image
        Y_train = label

    
    return (X_train, Y_train, X_test, Y_test)
        


if __name__ == "__main__":
    X_train, Y_train, X_test, Y_test = preprocess_data()
    print(X_test.shape)
    print(Y_test.shape)
    print(X_train.shape)
    print(Y_train.shape)