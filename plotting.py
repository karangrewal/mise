"""
Plotting images using Matplotlib
"""
import matplotlib.pyplot as plt
import numpy as np

from attacks import fgsm, targeted_fgsm, iterative_fgsm, random_noise
from mnist import load_data
from project import forward, loss, gradients, get_batch, evaluate
from project import BATCH_SIZE, INPUT_DIM, NUM_CLASSES, NUM_ITERATIONS

EPSILON = 0.07

def plot_image(x):
    """
    x: Array with dimensions (28,28)
    """
    plt.cla()
    img_plot = plt.imshow(x)
    plt.show()


if __name__ == "__main__":
    # Load classifier parameters
    W = np.load("./params/W.npy")
    b = np.load("./params/b.npy")

    # Load data
    train_X, train_Y, test_X, test_Y = load_data()

    # Generate a random batch on *test data*
    X, Y = get_batch(test_X, test_Y)

    # First compute gradients
    grad = gradients(W, b, X, Y)
    Y = np.argmax(Y, axis=1)

    # 0. original example (not an attack!)
    Y_hat_original = np.argmax(forward(W, b, X), axis=1)

    # 1. fast-gradient sign method (FGSM)
    X_fgsm = fgsm(X, grad["dX"], EPSILON)
    Y_hat_fgsm = np.argmax(forward(W, b, X_fgsm), axis=1)

    # Print adversarial examples that cause the classifier to change output
    diffs = list()
    for i in range(BATCH_SIZE):
        if Y_hat_original[i] != Y_hat_fgsm[i]:
            print("output on original: {}\noutput on adversar: {}\n".format(Y_hat_original[i], Y_hat_fgsm[i]))
            plot_image(X[i].reshape((28,28)))
            plot_image(X_fgsm[i].reshape((28,28)))
