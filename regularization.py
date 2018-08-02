"""
Multi-class classifier on MNIST with L1 or L2 regularization
"""
import os
import numpy as np
from mnist import load_data
from project import forward, get_batch, evaluate
from project import BATCH_SIZE, INPUT_DIM, NUM_CLASSES

ALPHA = 1. # Regularization parameter
L2_REGULARIZATION = True
L1_REGULARIZATION = False

# Global variables
LEARNING_RATE = 0.01
NUM_ITERATIONS = 200

def loss(W, b, X, Y):
    """ Compute cross-entropy loss """
    loss = 0.
    forward_prop = forward(W, b, X)
    for n in range(X.shape[0]):
        for c in range(NUM_CLASSES):
            loss += Y[n, c] * np.log(forward_prop[n, c])
    loss = -1. * loss
    for d in range(INPUT_DIM):
        for c in range(NUM_CLASSES):
            loss += (ALPHA / 2.) * np.square(W[c, d]) if L2_REGULARIZATION else ALPHA * np.abs(W[c, d])
    return loss


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
                dX[n, d] += 2 * ALPHA * W[c, d] if L2_REGULARIZATION else ALPHA * np.sign(W[c, d])
            db_nc = -1. * Y[n, c] / forward_prop[n, c]
            db_nc = db_nc * forward_prop[n, c] * (1. - forward_prop[n, c])
            db[c] += db_nc
    return {"dW":dW, "db":db, "dX":dX}


if __name__ == "__main__":
    assert L2_REGULARIZATION != L1_REGULARIZATION
    if L2_REGULARIZATION:
        print("MNIST classifier with L2 reg.")
    else:
        print("MNIST classifier with L1 reg.")

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

        if L2_REGULARIZATION:
            np.save("./params/W_L2.npy", W)
            np.save("./params/b_L2.npy", b)
        else:
            np.save("./params/W_L1.npy", W)
            np.save("./params/b_L1.npy", b)

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
            