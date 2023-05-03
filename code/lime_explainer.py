
import pickle
import tensorflow as tf
from model import TumorClassifier

from lime import lime_image
from skimage.segmentation import mark_boundaries
from matplotlib import pyplot as plt
import numpy as np


def LIME_explainer(model, image):
    """
    This function takes in a trained model and a preprocessed image 
    and generates LIME explanation images.
    """

    def image_and_mask(title, positive_only=True, num_features=5,
                       hide_rest=True):

        temp, mask = explanation.get_image_and_mask(
            explanation.top_labels[0], positive_only=positive_only,
            num_features=num_features, hide_rest=hide_rest)
        plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))
        plt.title(title)
        plt.show()

    explainer = lime_image.LimeImageExplainer()

    def new_predict_fn(images):
        # images = images[tf.newaxis, :]
        # images = images[:, tf.newaxis]

        # print(images.shape)
        # print()
        # print(images)

        # images = tf.image.rgb_to_grayscale(images)
        images = tf.expand_dims(tf.reduce_sum(images, 3), -1)

        # display_image = tf.squeeze(images)
        # plt.imshow(display_image, cmap='gray', vmin=0, vmax=255)
        # plt.show()

        print(images.shape)

        return model.predict(images)

    explanation = explainer.explain_instance(
        image, new_predict_fn, top_labels=5, hide_color=0,
        num_samples=1000)

    # The top 5 superpixels that are most positive towards the class with the
    # rest of the image hidden
    image_and_mask("Top 5 superpixels", positive_only=True, num_features=5,
                   hide_rest=True)

    # The top 5 superpixels with the rest of the image present
    image_and_mask("Top 5 with the rest of the image present",
                   positive_only=True, num_features=5, hide_rest=False)

    # The 'pros and cons' (pros in green, cons in red)
    image_and_mask("Pros(green) and Cons(red)",
                   positive_only=False, num_features=10, hide_rest=False)

    # Select the same class explained on the figures above.
    ind = explanation.top_labels[0]
    # Map each explanation weight to the corresponding superpixel
    dict_heatmap = dict(explanation.local_exp[ind])
    heatmap = np.vectorize(dict_heatmap.get)(explanation.segments)
    plt.imshow(heatmap, cmap='RdBu', vmin=-heatmap.max(), vmax=heatmap.max())
    plt.colorbar()
    plt.title("Map each explanation weight to the corresponding superpixel")
    plt.show()


# main
if __name__ == "__main__":

    # grab an image from the pickle
    with open("test.pickle", "rb") as file:
        X_test = tf.cast(pickle.load(file), dtype=tf.float64)
        Y_test = tf.cast(pickle.load(file), dtype=tf.int64)
    Y_test = tf.expand_dims(Y_test, 1)

    X_test_single = X_test[0][tf.newaxis, :]
    Y_test_single = Y_test[0][tf.newaxis, :]

    # load model
    model = TumorClassifier()
    model.build(X_test_single.shape)
    model.load_weights("model_weights.h5")

    # run lime
    X_test_single = tf.squeeze(X_test_single)
    LIME_explainer(model, X_test_single)