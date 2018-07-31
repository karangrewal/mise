"""
Multi-class classifier on MNIST
"""
import os
import numpy as np
from mnist import load_data
from plotting import plot_image

# Global variables
# You can modify these if you want
BATCH_SIZE = 32
EPSILON = 0.07 # Step size for adversarial examples
INPUT_DIM = 784
LEARNING_RATE = 0.01
NUM_CLASSES = 10
NUM_ITERATIONS = 200

def forward(W, b, X):
    """
    W: shape (10, 784)
    b: shape (10,)
    X: shape (num examples, 784)
    """
    Z = np.dot(X, W.T) + b
    Z = np.exp(Z) / np.tile(np.sum(np.exp(Z), axis=1), (NUM_CLASSES,1)).T
    return Z


def loss(W, b, X, Y):
    """ Compute cross-entropy loss """
    loss = 0.
    forward_prop = forward(W, b, X)
    for n in range(X.shape[0]):
        for c in range(NUM_CLASSES):
            loss += Y[n, c] * np.log(forward_prop[n, c])
    return -1. * loss


def gradients(W, b, X, Y):
    """
    W: shape (10, 784)
    b: shape (10,)
    X: shape (num examples, 784)
    Y: shape (num examples, 10)
    """
    forward_prop = forward(W, b, X)
    other = np.exp(np.dot(X, W.T) + b)
    dW, db, dX = np.zeros((NUM_CLASSES, INPUT_DIM)), np.zeros(NUM_CLASSES), np.zeros((BATCH_SIZE, INPUT_DIM))
    for n in range(BATCH_SIZE):
        for c in range(NUM_CLASSES):
            for d in range(INPUT_DIM):
                dW_ncd = -1. * Y[n, c] / forward_prop[n, c]
                dW_ncd = dW_ncd * X[n, d] * forward_prop[n, c] * (1. - forward_prop[n, c])
                dW[c, d] += dW_ncd
                dX_ncd = forward_prop[n, c] * (W[c, d] - np.sum(other * W[:, d], axis=1)[n] / np.sum(other, axis=1)[n])
                dX_ncd = dX_ncd * -1. * Y[n, c] / forward_prop[n, c]
                dX[n, d] += dX_ncd
            db_nc = -1. * Y[n, c] / forward_prop[n, c]
            db_nc = db_nc * forward_prop[n, c] * (1. - forward_prop[n, c])
            db[c] += db_nc
    return {"dW":dW, "db":db, "dX":dX}


def get_batch(full_X, full_Y):
    """
    Return a random batch from full_X, full_Y
    """
    indices = np.random.randint(0, len(full_X), BATCH_SIZE)
    X = full_X[indices,:]
    Y = full_Y[indices,:]
    return X, Y


def evaluate(Y, Y_hat):
    """
    Y: shape (num examples,)
    Y_hat: shape (num examples,)
    """
    return 100. * np.sum((Y == Y_hat)) / BATCH_SIZE


if __name__ == "__main__":
    # Initialize weights and bias
    W, b = np.random.randn(NUM_CLASSES, INPUT_DIM), np.random.randn(NUM_CLASSES)

    # Load data
    train_X, train_Y, test_X, test_Y = load_data()

    # Training
    for it in range(1, NUM_ITERATIONS + 1):
        # Generate training batch
        X, Y = get_batch(train_X, train_Y)

        # Compute gradients, update weights and bias
        grad = gradients(W, b, X, Y)
        W = W - LEARNING_RATE * grad["dW"]
        b = b - LEARNING_RATE * grad["db"]

        # Save updated parameters
        if not os.path.isdir("./params/"):
            os.mkdir("./params/")
        np.save("./params/W.npy", W)
        np.save("./params/b.npy", b)

        # Check model accuracy on test data
        if it % 5 == 0:
            scores, losses = list(), list()
            for n in range(10):
                X, Y = get_batch(test_X, test_Y)
                Y_out = np.argmax(forward(W, b, X), axis=1)
                
                # Compute loss on X
                losses.append(loss(W, b, X, Y))
                
                # Compute accuracy on test data
                Y = np.argmax(Y, axis=1)
                scores.append(evaluate(Y, Y_out))
                
            print("Iter {}\tAccuracy {}%\t Loss {}".format(it, np.round(np.mean(scores)), np.mean(losses)))
            