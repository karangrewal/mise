"""
Fast gradient sign method (FGSM) on classifier
"""
import numpy as np
from mnist import load_data

BATCH_SIZE = 32
EPSILON = 0.05 # Step size for adversarial examples
INPUT_DIM = 784
LEARNING_RATE = 0.01
NUM_CLASSES = 10
NUM_ITERATIONS = 5000

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
    dW, db = np.zeros((NUM_CLASSES, INPUT_DIM)), np.zeros(NUM_CLASSES)
    for n in range(X.shape[0]):
        for c in range(NUM_CLASSES):
            for d in range(INPUT_DIM):
                dW_ncd = -1. * Y[n, c] / forward_prop[n, c]
                dW_ncd = dW_ncd * X[n, d] * forward_prop[n, c] * (1. - forward_prop[n, c])
                dW[c, d] += dW_ncd
            db_nc = -1. * Y[n, c] / forward_prop[n, c]
            db_nc = db_nc * forward_prop[n, c] * (1. - forward_prop[n, c])
            db[c] += db_nc
    return {"dW":dW, "db":db}

def adversarial_examples(W, b, X, Y):
    """ Compute adversarial examples using FGSM """
    forward_prop = forward(W, b, X)
    other = np.exp(np.dot(X, W.T) + b)
    dX = np.zeros((BATCH_SIZE, INPUT_DIM))
    for n in range(BATCH_SIZE):
        for c in range(NUM_CLASSES):
            for d in range(INPUT_DIM):
                dX_ncd = forward_prop[n, c] * (W[c, d] - np.sum(other * W[:, d], axis=1)[n] / np.sum(other, axis=1)[n])
                dX_ncd = dX_ncd * -1. * Y[n, c] / forward_prop[n, c]
                dX[n, d] += dX_ncd
    return {"dX":dX}

if __name__ == "__main__":
    # Initialize weights and bias
    W, b = np.random.randn(NUM_CLASSES, INPUT_DIM), np.random.randn(NUM_CLASSES)

    # Load data
    train_X, train_Y, test_X, test_Y = load_data()

    # Training
    for it in range(1, NUM_ITERATIONS + 1):
        # Pick a random sample
        indices = np.random.randint(0, len(train_X), size=BATCH_SIZE)
        X = train_X[indices,:]
        Y = train_Y[indices,:]

        # Compute gradients, update weights and bias
        # YOUR CODE HERE

        # Check model accuracy
        if it % 5 == 0:
            scores, losses = list(), list()
            scores_adv, losses_adv = list(), list()
            for n in range(10):
                indices = np.random.randint(0, len(train_X), size=BATCH_SIZE)
                X, Y = train_X[indices,:], train_Y[indices,:]
                
                # Create adversarial example and evaluate model on that
                # Hint: You need a variable called `X_adv`
                # YOUR CODE HERE

                Y_out = np.argmax(forward(W, b, X), axis=1)
                
                # Get output of classifier on adversarial examples
                # Hint 1: You need to create a variable called `Y_out_adv`
                # Hint 2: Use `np.argmax`
                # YOUR CODE HERE
                
                # Compute loss on X and X_adv
                losses.append(loss(W, b, X, Y))
                losses_adv.append(loss(W, b, X_adv, Y))

                Y = np.argmax(Y, axis=1)
                scores.append(100. * np.sum((Y == Y_out)) / BATCH_SIZE)
                scores_adv.append(100. * np.sum((Y == Y_out_adv)) / BATCH_SIZE)

            print("Iter {} Regular\tAccuracy {}%\t Loss {}".format(it, np.round(np.mean(scores)), np.mean(losses)))
            print("Iter {} Advers.\tAccuracy {}%\t Loss {}".format(it, np.round(np.mean(scores_adv)), np.mean(losses_adv)))
            print("\n")
