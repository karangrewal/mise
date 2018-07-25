"""
Linear 2-class classification
"""
import numpy as np

BATCH_SIZE = 32
INPUT_DIM = 10
LEARNING_RATE = 0.001
NUM_ITERATIONS = 50

def forward(w, b, X):
    return 1. / (1. + np.exp(-np.dot(X, w) - np.tile(b, (32)), dtype=np.float64))

def gradient(w, b, X, Y):
    forward_prop = forward(w, b, X)
    dw, db = np.zeros((INPUT_DIM)), 0.
    for n in range(X.shape[0]):
        for d in range(INPUT_DIM):
            dw[d] += (1. - forward_prop[n]) * X[n, d] * Y[n, 1] + forward_prop[n] * X[n, d] * (1. - Y[n, 1])
        db += (1. - forward_prop[n]) * Y[n, 1] + forward_prop[n] * (1. - Y[n, 1])
    return {"dw":dw, "db":db}

def get_batch():
    X_a = np.random.normal(1., 1., (BATCH_SIZE / 2, INPUT_DIM))
    X_b = np.random.normal(-1., 1., (BATCH_SIZE / 2, INPUT_DIM))
    X = np.concatenate((X_a, X_b), axis=0)
    Y_top = np.concatenate((np.ones((BATCH_SIZE / 2, 1)), np.zeros((BATCH_SIZE / 2, 1))), axis=1)
    Y_bottom = np.concatenate((np.zeros((BATCH_SIZE / 2, 1)), np.ones((BATCH_SIZE / 2, 1))), axis=1)
    Y = np.concatenate((Y_top, Y_bottom), axis=0)
    return X, Y

if __name__ == "__main__":
    w = np.random.randn(INPUT_DIM)
    b = np.random.randn()

    # Training
    for it in range(NUM_ITERATIONS):
        X, Y = get_batch()
        grad = gradient(w, b, X, Y)
        w = w - LEARNING_RATE * grad["dw"]
        b = b - LEARNING_RATE * grad["db"]
        
        # Validate
        # if it % 25 == 0:
        if True:
            scores = list()
            for n in range(10):
                X, Y = get_batch()
                Y_out = np.round(forward(w, b, X))
                Y = 1. - np.argmax(Y, axis=1)
                # if n == 3:
                #     print(np.concatenate((Y.reshape(-1,1), Y_out.reshape(-1,1)), axis=1))
                scores.append(100. * np.sum((Y == Y_out)) / BATCH_SIZE)
            print("Iter {}: Accuracy {}%".format(it, np.mean(scores)))
