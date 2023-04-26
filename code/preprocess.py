import numpy as np
import tensorflow as tf
import pickle 
import cv2
import tensorflow_addons as tfa
from pathlib import Path

def preprocess_data():

    # collect datasets
    path_test = Path("../data/Testing")
    path_train = Path("../data/Training")

    test_dataset = tf.keras.utils.image_dataset_from_directory(
        path_test, 
        color_mode="grayscale",
        batch_size=10000,
        shuffle=False,
        image_size=(112, 112),
    )
    
    train_dataset = tf.keras.utils.image_dataset_from_directory(
        path_train, 
        color_mode="grayscale",
        batch_size=10000,
        shuffle=False,
        image_size=(112, 112),
    )

    X_test, Y_test, X_train, Y_train = ..., ..., ..., ...

    for _, (X_batch, Y_batch) in enumerate(test_dataset):
        X_test = X_batch
        Y_test = Y_batch
    
    for _, (X_batch, Y_batch) in enumerate(train_dataset):
        X_train = X_batch
        Y_train = Y_batch
    

    # gaussian blurring
    X_test = tfa.image.gaussian_filter2d(X_test, filter_shape = (5, 5))
    X_train = tfa.image.gaussian_filter2d(X_train, filter_shape = (5, 5))
    
    # histogram equalization to adjust contrast
    X_train = histogram_equalizer(X_train)
    X_test = histogram_equalizer(X_test)

    X_train /= 255
    X_test /= 255

    return (X_train, Y_train, X_test, Y_test)
        
def histogram_equalizer(img_list):
    img_list = img_list.numpy().astype(np.uint8)    
    new_img_list = np.zeros_like(img_list)                                     
    for i, img in enumerate(img_list):
        new_img_list[i] = tf.expand_dims(cv2.equalizeHist(img),-1)

    return tf.convert_to_tensor(new_img_list)

if __name__ == "__main__":

    X_train, Y_train, X_test, Y_test = preprocess_data()
    print("X_train min", np.min(X_train))
    print("Y_train min", np.min(Y_train))
    print("X_train max", np.max(X_train))
    print("Y_train max", np.max(Y_train))
    print(type(X_train))
    print(type(Y_train))
    print(type(X_test))
    print(type(Y_test))
    with open('train.pickle', 'wb') as file:
        pickle.dump(X_train, file)
        pickle.dump(Y_train, file)
    with open('test.pickle', 'wb') as file:
        pickle.dump(X_test, file)
        pickle.dump(Y_test, file)

    print(f"\n\n\t\t --- Pickling file!! --- \t\t\n\n")    
    print(f"X_Test shape: {X_test.shape}")
    print(f"Y_Test shape: {Y_test.shape}")
    print(f"X_Train shape: {X_train.shape}")
    print(f"Y_Train shape: {Y_train.shape}")