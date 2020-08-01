import numpy as np
import matplotlib.pyplot as plt


def emperical_distribution(x, data):
    """ Adds data and normalizes, between 0 and 1.
        INPUT:
            x:
                list, array or dataframe of floats or ints
            data:
                list, array or dataframe of floats or ints
            Same length required
        OUTPUT:
            New output list between 0 and 1 of length len(x)
    """
    weight = 1.0 / len(data)
    count = np.zeros(shape=len(x))
    for datum in data:
        count = count + np.array(x >= datum)
    return weight * count


def plot_emperical_distribution(ax, data):
    """ Plots a emperical CMF of data on the matplotib axis ax.
    INPUT:
        ax:
            matplotlib axis
            (use 'fig, ax, subplots(1,1)')
        data:
            list, array or dataframe of floats or ints
        Same length required
    OUTPUT:
        A CMF plot.
    """
    if (type(data).__name__ == 'DataFrame'):
        for column in data:
            minimum = data[column].min()
            maximum = data[column].max()
            buff = (maximum - minimum) / 10
            line = np.linspace(data[column].min() - buff,
                               data[column].max() + buff, len(data[column]))
            ax.plot(line, emperical_distribution(line, data[column]))
    else:
        minimum = min(data)
        maximum = max(data)
        buff = (maximum - minimum) / 10
        line = np.linspace(minimum - buff, maximum + buff, len(data))
        ax.plot(line, emperical_distribution(line, data))
    return None
