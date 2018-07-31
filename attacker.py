"""
Perform attacks on classifier
"""
import numpy as np

from attacks import fgsm, targeted_fgsm, iterative_fgsm, random_noise
from mnist import load_data
from project import forward, loss, gradients
from project import BATCH_SIZE, EPSILON, INPUT_DIM, NUM_CLASSES, NUM_ITERATIONS


def generate_false_labels(Y):
    """
    Y: shape (num examples, 10)
    """
    # YOUR CODE HERE
    # You need to return a matrix with the same dimensions as Y


def evaluate(Y, Y_hat):
    """
    Y: shape (num examples,)
    Y_hat: shape (num examples,)
    """
    # YOUR CODE HERE
    # You need to return a float that represents a percentage value


if __name__ == "__main__":
    # Load classifier parameters
    W = np.load("./params/W.npy")
    b = np.load("./params/b.npy")

    # Load data
    train_X, train_Y, test_X, test_Y = load_data()

    # Perform adversarial attacks; for each of these, you should also keep 
    # score of the classifier's accuracy during each type of attack to compare
    # afterwards

    # 0. original example (not an attack!)
    # YOUR CODE HERE


    print("[original]\tAccuracy:")

    # 1. fast-gradient sign method (FGSM)
    # YOUR CODE HERE


    print("[FGSM]\tAccuracy:")

    # 2. targeted fast-gradient sign method (T-FGSM)
    # YOUR CODE HERE


    print("[T-FGSM]\tAccuracy:")

    # 3. iterative fast-gradient sign method (I-FGSM)
    # YOUR CODE HERE


    print("[I-FGSM]\tAccuracy:")

    # 4. random noise
    # YOUR CODE HERE


    print("[noise]\tAccuracy:")
