"""
Different attacks for generating adversarial examples
"""
import numpy as np


############################## WHITE BOX ATTACKS ##############################

def fgsm(X, dX, step_size=0.07):
    """
    X: the original batch of data
    dX: gradient of the loss function with respect to X
    step_size: the step size
    """
    # YOUR CODE HERE
    # You should return the batch of adversarial examples


def targeted_fgsm(X, dX, step_size=0.07):
    """
    X: the original batch of data
    dX: gradient of the *wrong* loss function with respect to X
    step_size: the step size
    """
    # YOUR CODE HERE
    # You should return the batch of adversarial examples


def iterative_fgsm(X, dX, T=10, step_size=0.07):
    """
    X: the original batch of data
    dX: gradient of the loss function with respect to X
    T: number of steps
    step_size: the step size
    """
    # YOUR CODE HERE
    # You should return the batch of adversarial examples
    # HINT: you need to use a loop ;)


############################## WHITE BOX ATTACKS ##############################

def random_noise(X, step_size=0.07):
    """
    X: the original batch of data
    step_size: the step size
    """
    # YOUR CODE HERE
    # You should return the batch of adversarial examples
    # HINT: use np.random.randn

    