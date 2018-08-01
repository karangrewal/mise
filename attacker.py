"""
Perform attacks on classifier
"""
import numpy as np

from attacks import fgsm, targeted_fgsm, iterative_fgsm, random_noise
from mnist import load_data
from project import forward, loss, gradients, get_batch, evaluate
from project import BATCH_SIZE, EPSILON, INPUT_DIM, NUM_CLASSES, NUM_ITERATIONS


def generate_false_labels(Y):
    """
    Y: shape (num examples, 10)
    """
    return np.random.randint(0, NUM_CLASSES, Y.shape[0])


if __name__ == "__main__":
    # Load classifier parameters
    W = np.load("./params/W.npy")
    b = np.load("./params/b.npy")

    # Load data
    train_X, train_Y, test_X, test_Y = load_data()

    # Generate a random batch on *test data*
    X, Y = get_batch(test_X, test_Y)

    # Perform adversarial attacks; for each of these, you should also keep 
    # score of the classifier's accuracy during each type of attack to compare
    # afterwards

    # First compute gradients
    grad = gradients(W, b, X, Y)

    Y = np.argmax(Y, axis=1)

    # 0. original example (not an attack!)
    Y_hat_original = np.argmax(forward(W, b, X), axis=1)
    score = evaluate(Y, Y_hat_original)
    print("[original]\tAccuracy {}%".format(score))
    print(Y_hat_original)

    # 1. fast-gradient sign method (FGSM)
    X_fgsm = fgsm(X, grad["dX"], 2*EPSILON)
    Y_hat_fgsm = np.argmax(forward(W, b, X_fgsm), axis=1)
    score = evaluate(Y, Y_hat_fgsm)
    print("[  FGSM]\tAccuracy {}%".format(score))
    print(Y_hat_fgsm)

    # 2. targeted fast-gradient sign method (T-FGSM)
    Y_false = generate_false_labels(Y)
    X_tfgsm = targeted_fgsm(X, grad["dX"], 2*EPSILON)
    Y_hat_tfgsm = np.argmax(forward(W, b, X_tfgsm), axis=1)
    score = evaluate(Y, Y_hat_tfgsm)
    print("[T-FGSM]\tAccuracy {}%".format(score))
    print(Y_hat_tfgsm)

    # 3. iterative fast-gradient sign method (I-FGSM)
    X_ifgsm = iterative_fgsm(X, grad["dX"], 10, 2*EPSILON)[-1]
    Y_hat_ifgsm = np.argmax(forward(W, b, X_ifgsm), axis=1)
    score = evaluate(Y, Y_hat_ifgsm)
    print("[I-FGSM]\tAccuracy {}%".format(score))
    print(Y_hat_ifgsm)

    # 4. random noise
    X_noise = random_noise(X, 2*EPSILON)
    Y_hat_noise = np.argmax(forward(W, b, X_ifgsm), axis=1)
    score = evaluate(Y, Y_hat_noise)
    print("[ noise]\tAccuracy {}%".format(score))
    print(Y_hat_noise)

