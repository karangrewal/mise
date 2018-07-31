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
    return X + step_size * dX
    

def targeted_fgsm(X, dX, step_size=0.07):
    """
    X: the original batch of data
    dX: gradient of the *wrong* loss function with respect to X
    step_size: the step size
    """
    return X + step_size * dX


def iterative_fgsm(X, dX, T=10, step_size=0.07):
    """
    X: the original batch of data
    dX: gradient of the loss function with respect to X
    T: number of steps
    step_size: the step size
    """
    examples = list()
    X_current = X
    for t in range(T):
        X_current = X_current + (step_size / T) * dX
        examples.append(X_current)
    return examples


############################## BLACK BOX ATTACKS ##############################

def random_noise(X, step_size=0.07):
    """
    X: the original batch of data
    step_size: the step size
    """
    return X + step_size * np.randn(X.shape[0], X.shape[1])

