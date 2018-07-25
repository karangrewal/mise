"""
Linear 2-class classification

This classifier distinguishes between samples drawn from different normal
distributions
"""
import numpy as np

BATCH_SIZE = 16
INPUT_DIM = 10
LEARNING_RATE = 0.01
NUM_ITERATIONS = 50


def forward(w, b, X):
    # Compute 
    # Hint: Look up the functions np.exp, np.dot, np.tile, they will be useful
    # YOUR CODE HERE

########## YOU DON'T NEED TO KNOW HOW THESE WORK, BUT THEY WORK! ;) ###########

def gradient(w, b, X, Y):
    """
    Compute the gradient of the cross-entropy loss and return a dictionary
    that contains the gradients
    """
    forward_prop = forward(w, b, X)
    dw, db = np.zeros((INPUT_DIM)), 0.
    for n in range(X.shape[0]):
        for d in range(INPUT_DIM):
            dw[d] += (1. - forward_prop[n]) * X[n, d] * Y[n, 0] + forward_prop[n] * X[n, d] * (1. - Y[n, 0])
        db += (1. - forward_prop[n]) * Y[n, 0] + forward_prop[n] * (1. - Y[n, 0])
    return {"dw":dw, "db":db}

def get_batch():
    """
    Draw samples from two normal distributions with means 1 and -1
    """
    X_a = np.random.normal(1., 1.3, (BATCH_SIZE / 2, INPUT_DIM))
    X_b = np.random.normal(-1., 1.3, (BATCH_SIZE / 2, INPUT_DIM))
    X = np.concatenate((X_a, X_b), axis=0)
    Y_top = np.concatenate((np.ones((BATCH_SIZE / 2, 1)), np.zeros((BATCH_SIZE / 2, 1))), axis=1)
    Y_bottom = np.concatenate((np.zeros((BATCH_SIZE / 2, 1)), np.ones((BATCH_SIZE / 2, 1))), axis=1)
    Y = np.concatenate((Y_top, Y_bottom), axis=0)
    return X, Y

###############################################################################

if __name__ == "__main__":
    # Initialize weights and bias randomly
    w = np.random.randn(INPUT_DIM)
    b = np.random.randn()

    # Training
    for it in range(NUM_ITERATIONS):
        # Get a "batch" of examples
        X, Y = get_batch()

        # Compute gradient, then update weights and bias
        # YOUR CODE HERE
        
        # Check accuracy of our model
        scores = list()
        for n in range(10):
            X, Y = get_batch()
            Y_out = np.round(forward(w, b, X))
            Y = np.argmax(Y, axis=1)
            scores.append(100. * np.sum((Y == Y_out)) / BATCH_SIZE)
        print("Iter {}: Accuracy {}%".format(it, np.mean(scores)))
