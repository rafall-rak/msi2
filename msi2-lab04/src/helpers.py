from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap


class Classifier(ABC):
    @abstractmethod
    def fit(self, X, y):
        """
        Fit the classifier to the given data.

        Parameters
        ----------
        X : array_like
            The feature data, shape (n_samples, n_features)
        y : array_like
            The labels, shape (n_samples,)

        Returns
        -------
        self
        """
        pass

    @abstractmethod
    def predict(self, X):
        """
        Predict the class label(s) given some input data.

        Parameters
        ----------
        X : array_like
            The input data, shape (n_samples, n_features)

        Returns
        -------
        y : array_like
            The predicted class labels, shape (n_samples,)

        """
        pass


def plot_decision_regions(X, y, clf: Classifier, resolution=0.02):
    """
    Plot the decision regions for a classifier.

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
        The input data.
    y : ndarray, shape (n_samples,)
        The target values.
    clf : object
        A trained classifier.
    resolution : float, optional
        The resolution of the grid. Defaults to 0.02.

    Returns
    -------
    None

    """
    markers = ('o', 's', '^', 'v', '<')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    lab = clf.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    lab = lab.reshape(xx1.shape)
    plt.contourf(xx1, xx2, lab, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    label=f'Class {cl}',
                    edgecolor='black')
