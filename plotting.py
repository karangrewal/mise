"""
Plotting images using Matplotlib

Some useful code for implementing the function:
>>> train_X, train_Y, test_X, test_Y = load_data()
>>> fig_to_plot = train_X[12, :]
>>> fig_to_plot = fig_to_plot.reshape((28,28))
>>> img_plot = plt.imshow(fig_to_plot)
>>> plt.show()
"""
import matplotlib.pyplot as plt
import numpy as np

def plot_image(x):
    """
    x: Array with dimensions (28,28)
    """
    # YOUR CODE HERE
    # HINT: you don't need to return anything