import numpy as np
from mnist import load_data

BATCH_SIZE = 32
INPUT_DIM = 784
LEARNING_RATE = 0.01
NUM_CLASSES = 10
NUM_ITERATIONS = 5000

def forward(W, b, X):
    """ X has shape (num examples, 784) """
    Z = np.matmul(X, W) + np.tile(b, (BATCH_SIZE, 1))
    Z = np.exp(Z) / np.tile(np.sum(np.exp(Z), axis=1), (10,1)).T
    return Z

def gradients(W, b, X, Y):
    """
    W: shape (784, 10)
    b: shape (10,)
    X: shape (784, num examples)
    Y: shape (1, num examples)
    """
    A = np.exp(np.matmul(X, W) + np.tile(b, (BATCH_SIZE,1)))
    dw = np.tile(np.sum(A, axis=0), (BATCH_SIZE, 1)) - A
    dw = dw / np.tile(np.sum(A, axis=0), (BATCH_SIZE, 1))
    dw = np.matmul(X.T, dw)
    db = np.tile(np.sum(A, axis=0), (BATCH_SIZE, 1)) - A
    db = db / np.tile(np.sum(A, axis=0), (BATCH_SIZE, 1))
    db = np.mean(db, axis=0)
    return {"dw":dw, "db":db}

def gradients2(W, b, X, Y):
    """
    W: shape (784, 10)
    b: shape (10,)
    X: shape (784, num examples)
    Y: shape (1, num examples)
    """
    dw = np.zeros((INPUT_DIM, NUM_CLASSES))
    for j in range(INPUT_DIM):
        for i in range(NUM_CLASSES):
            dw[j,i] = -1. * np.sum(Y, axis=1)[i] * np.sum(X, axis=1)[i] * (1. - )
    return {"dw":dw, "db":db}

def optimize(W, b, X, Y):
    """ Optimization over batch (X,Y) """
    grads = gradients(W, b, X, Y)
    W = W - LEARNING_RATE * grads["dw"]
    # print(b.shape, grads["db"].shape)
    b = b - LEARNING_RATE * grads["db"]
    return W, b

if __name__ == "__main__":
    # Initialize logistic regression model
    W, b = np.random.randn(INPUT_DIM, NUM_CLASSES), np.random.randn(NUM_CLASSES)

    # Load data
    train_X, train_Y, test_X, test_Y = load_data()

    # Training
    for it in range(NUM_ITERATIONS):
        indices = np.random.randint(0, len(train_X), size=BATCH_SIZE)
        batch_X = train_X[indices,:]
        batch_Y = train_Y[indices,:]
        W, b = optimize(W, b, batch_X, batch_Y)

        # Validate
        if it % 25 == 0:
            scores = list()
            for n in range(10):
                indices = np.random.randint(0, len(train_X), size=BATCH_SIZE)
                batch_X = train_X[indices,:]
                batch_Y = train_Y[indices,:]
                Y_out = np.argmax(forward(W, b, batch_X), axis=1)
                batch_Y = np.argmax(batch_Y, axis=1)
                scores.append(100. * np.sum((batch_Y == Y_out)) / BATCH_SIZE)
            print("Iter {}: Accuracy {}%".format(it, np.mean(scores)))
