import numpy as np
import matplotlib.pyplot as plt


def graph_series():
    '''
    This is more for example than to be used as a function call.
    '''
    fig, axes = plt.subplots(nrows=5, ncols=5, 
                         sharex=True, sharey=True, 
                         figsize=(12, 8))

    for i, ax in enumerate(axes.flatten()):
        ax.plot(xs, ys, 'o', color='grey', 
                markeredgewidth=0, alpha=0.5, ms=2)
        l1 = ax.plot(x, y, color = 'black')
        # increase by 15 out of a total 500
        l2 = ax.plot(x, models[15*i], color="purple", linewidth=2)
        ax.text(-2.8, 0.8, str(15*i))
    
    axes[0, 0].set_xlim(-np.pi-pad, np.pi+pad)
    axes[0, 0].set_ylim([-1.5, 1.5])    
    fig.suptitle("Boosting Statges Over Time", fontsize=20)


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
