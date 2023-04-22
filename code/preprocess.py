import numpy as np
import tensorflow as tf
import pickle 
import cv2
import tensorflow_addons as tfa
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

    # Gaussian blurring
    X_test = tfa.image.gaussian_filter2d(
        X_test,
        filter_shape = (5, 5),
    )

    # # histogram equalization to adjust contrast
    X_train = histogram_equal(X_train)
    X_test = histogram_equal(X_test)

    return (X_train, Y_train, X_test, Y_test)
        
def histogram_equal(img_list):
    img_list = img_list.numpy().astype(np.uint8)    
    new_img_list = np.zeros_like(img_list)                                     
    for i, img in enumerate(img_list):
        new_img_list[i] = tf.expand_dims(cv2.equalizeHist(img),-1)

    return tf.convert_to_tensor(new_img_list)



if __name__ == "__main__":
    X_train, Y_train, X_test, Y_test = preprocess_data()
    with open('train.pickle', 'wb') as file:
        pickle.dump(X_train, file)
        pickle.dump(Y_train, file)
    with open('test.pickle', 'wb') as file:
        pickle.dump(X_test, file)
        pickle.dump(Y_test, file)
    # print(X_test.shape)
    # print(Y_test.shape)
    # print(X_train.shape)
    # print(Y_train.shape)