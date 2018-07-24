"""
maximize the function f(x) = -(x-1)^2 - (y+3)^2
"""
import numpy as np

STEPSIZE = 0.1

def f(x, y):
    return -1 * ((x-1)**2 + (y+3)**2)

def gradient(x, y):
    # YOUR CODE HERE

if __name__ == "__main__":
    # Initialize random point
    x_0 = np.random.randint(-10, -10)
    y_0 = np.random.randint(-10, -10)

    for n in range(50):
        print("Iteration {}: {}".format(n), f(x_0, y_0))

        # Take step in direction of gradient
        # YOUR CODE HERE
