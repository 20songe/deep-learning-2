import pickle
import tensorflow as tf
import numpy as np
from tqdm import tqdm
from model import TumorClassifier
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib as plt

def train(model, train_inputs, train_labels):

    loss_list = []

    x = tf.image.random_flip_left_right(train_inputs, seed=42)
    # x = tf.image.random_flip_up_down(x, seed=42)
    # x = tf.image.random_contrast(x, 0.75, 1.25)

    indices = np.arange(len(x))
    indices = tf.random.shuffle(indices, seed=42)

    x = tf.gather(x, indices)
    y = tf.gather(train_labels, indices)
    
    batch_size = 128
    examples = x.shape[0]

    pbar = tqdm(total=examples // batch_size, position=0, leave=True, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} ')
    for i in range(examples // batch_size):
        input_batch = x[i * batch_size:(i+1)*batch_size,:]
        label_batch = y[i * batch_size:(i+1)*batch_size,:]
        # print(label_batch.shape)
        # print("label_batch", label_batch)

        with tf.GradientTape() as tape:
            logits = model.call(input_batch, is_training=True)
            loss = model.loss(tf.reshape(label_batch,-1), logits)
            # print("log", logits.shape)

        grads = tape.gradient(loss, model.trainable_weights)
        model.optimizer.apply_gradients(zip(grads, model.trainable_weights))

        loss_list.append(loss)

        pbar.set_description("Training loss for Batch %s: %.4f" % (int(i), float(loss)))
        pbar.update()
    
    return loss_list

def test(model, test_inputs, test_labels):
    logits = model.call(test_inputs, is_training=False)
    # print(logits)
    # print(tf.argmax(logits, 1))
    return model.accuracy(logits, test_labels)

if __name__ == "__main__":
    with open("train.pickle", "rb") as file:
        X_train = tf.cast(pickle.load(file), dtype=tf.float64)
        Y_train = tf.cast(pickle.load(file), dtype=tf.int64)

    with open("test.pickle", "rb") as file:
        X_test = tf.cast(pickle.load(file), dtype=tf.float64)
        Y_test = tf.cast(pickle.load(file), dtype=tf.int64)
    
    Y_train = tf.expand_dims(Y_train, 1)
    Y_test = tf.expand_dims(Y_test, 1)

    model = TumorClassifier()
    # model.compile(
    #     optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    #     loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    # )
    # model.build(X_test.shape)
    # print(model.summary())

    NUM_EPOCHS = 100
    
    for i in range(NUM_EPOCHS):
        print("Epoch: ", i + 1)
        loss_list = train(model, X_train, Y_train)
        # print(loss_list)
        test_acc = test(model, X_test, Y_test).numpy()
        print("Test Accuracy: ", test_acc)

    # Saves Model Weights to h5 file
    model.save_weights('model_weights.h5')

    

    # # Assuming predicted and true values are in arrays pred and true respectively
    # confusion = confusion_matrix(Y_test, np.argmax(model(X_test, False),1))

    # # Visualize the confusion matrix as a heatmap
    # sns.heatmap(confusion, annot=True, fmt='d')
    # plt.xlabel('Predicted')
    # plt.ylabel('True')
    # plt.show()
    # plt.savefig("confusion_matrix.png")