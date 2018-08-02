"""
Multi-class classifier on MNIST with adversarial training
"""
import os
import numpy as np
from attacker import generate_false_labels
from attacks import fgsm, targeted_fgsm, iterative_fgsm, random_noise
from mnist import load_data
from project import forward, loss, gradients, get_batch, evaluate
from project import BATCH_SIZE, INPUT_DIM, NUM_CLASSES

# Global variables
EPSILON = 0.07
LEARNING_RATE = 0.01
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


if __name__ == "__main__":
    # Initialize weights and bias
    W, b = np.random.randn(NUM_CLASSES, INPUT_DIM), np.random.randn(NUM_CLASSES)

    # Load data
    train_X, train_Y, test_X, test_Y = load_data()

    # Training
    for it in range(1, NUM_ITERATIONS + 1):
        # Generate training batch
        X_original, Y_original = get_batch(train_X, train_Y)

        # Generate adversarial examples by FGSM and T-FGSM
        grad = gradients(W, b, X_original, Y_original)
        X = np.concatenate((X_original, fgsm(X_original, grad["dX"], EPSILON)), axis=0)
        Y = np.concatenate((Y_original, Y_original), axis=0)

        Y_false = generate_false_labels(Y_original)
        Y_false = np.eye(NUM_CLASSES)[Y_false]
        grad = gradients(W, b, X_original, Y_false)
        X = np.concatenate((X, targeted_fgsm(X_original, grad["dX"], EPSILON)))
        Y = np.concatenate((Y, Y_original), axis=0)

        indices = np.random.randint(0, X.shape[0], BATCH_SIZE)
        X = X[indices,:]
        Y = Y[indices,:]

        # Compute gradients, update weights and bias
        grad = gradients(W, b, X, Y)
        W = W - LEARNING_RATE * grad["dW"]
        b = b - LEARNING_RATE * grad["db"]

        # Save updated parameters
        if not os.path.isdir("./params/"):
            os.mkdir("./params/")
        np.save("./params/W_adv_training.npy", W)
        np.save("./params/b_adv_training.npy", b)

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
            