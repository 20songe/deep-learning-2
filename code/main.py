import pickle
import tensorflow as tf
import numpy as np
from tqdm import tqdm
from model import TumorClassifier

def train(model, train_inputs, train_labels):

    loss_list = []

    x = tf.image.random_flip_left_right(train_inputs, seed=42)

    indices = np.arange(len(x))
    indices = tf.random.shuffle(indices, seed=42)
    x = tf.gather(x, indices)
    y = tf.gather(train_labels, indices)
    
    batch_size = 32
    examples = len(x)

    pbar = tqdm(total=examples // batch_size, position=0, leave=True, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} ')
    for i in range(examples // batch_size):
        input_batch = x[i * batch_size:(i+1)*batch_size,:]
        label_batch = y[i * batch_size:(i+1)*batch_size,:]

        with tf.GradientTape() as tape:
            logits = model(input_batch, is_training=True)
            loss = model.loss(label_batch, logits)

        grads = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(grads, model.trainable_variables))

        loss_list.append(loss)

        pbar.set_description("Training loss for step %s: %.4f" % (int(i), float(loss)))
        pbar.update()
    
    return loss_list

def test(model, test_inputs, test_labels):
    logits = model.call(test_inputs, is_training=False)
    print(logits)
    print(tf.argmax(logits, 1))
    return model.accuracy(logits, test_labels)

if __name__ == "__main__":
    with open("train.pickle", "rb") as file:
        X_train = tf.cast(pickle.load(file), dtype=tf.int32)
        Y_train = tf.cast(pickle.load(file), dtype=tf.int32)

    with open("test.pickle", "rb") as file:
        X_test = tf.cast(pickle.load(file), dtype=tf.int32)
        Y_test = tf.cast(pickle.load(file), dtype=tf.int32)
    
    Y_train = tf.expand_dims(Y_train, 1)
    Y_test = tf.expand_dims(Y_test, 1)

    model = TumorClassifier()
    model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=1e-3),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    )
    # model.build(X_test.shape)
    # print(model.summary())

    NUM_EPOCHS = 25
    
    for _ in range(NUM_EPOCHS):
        loss_list = train(model, X_train, Y_train)
        # print(loss_list)
        # train_acc = test(model, X_train, Y_train)
        test_acc = test(model, X_test, Y_test)
        # print("Training Accuracy: ", train_acc)
        print("Test Accuracy: ", test_acc)
