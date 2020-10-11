import numpy as np
import matplotlib.pyplot as plt

class DataBinWrapper(object):

    def __init__(self, max_bins=10):
        self.max_bins = max_bins
        self.x_range_map = None

    def fit(self, X):
        n_sample, n_feature = X.shape
        self.x_range_map = [[] for _ in range(0, n_feature)]
        for index in range(0, n_feature):
            tmp = sorted(X[:, index])
            for percent in range(1, self.max_bins):
                percent_value = np.percentile(tmp, (1.0 * percent / self.max_bins) * 100 // 1)
                self.x_range_map[index].append(percent_value)
            self.x_range_map[index] = sorted(list(set(self.x_range_map[index])))

    def transform(self, X):
        if X.ndim == 1:
            return np.asarray([np.digitize(X[i], self.x_range_map[i]) for i in range(X.size)])
        else:
            return np.asarray([np.digitize(X[:, i], self.x_range_map[i]) for i in range(X.shape[1])]).T



def softmax(x):
    if x.ndim == 1:
        return np.exp(x) / np.exp(x).sum()
    else:
        return np.exp(x) / np.exp(x).sum(axis=1, keepdims=True)

def plot_decision_function(X, y, clf, support_vectors=None):
    plot_step = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                         np.arange(y_min, y_max, plot_step))

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.4)
    plt.scatter(X[:, 0], X[:, 1], alpha=0.8, c=y, edgecolor='k')
    # 绘制支持向量
    if support_vectors is not None:
        plt.scatter(X[support_vectors, 0], X[support_vectors, 1], s=80, c='none', alpha=0.7, edgecolor='red')


def plot_contourf(data, func, lines=3):
    n = 256
    x = np.linspace(data[:, 0].min(), data[:, 0].max(), n)
    y = np.linspace(data[:, 1].min(), data[:, 1].max(), n)
    X, Y = np.meshgrid(x, y)
    C = plt.contour(X, Y, func(np.c_[X.reshape(-1), Y.reshape(-1)]).reshape(X.shape), lines, colors='g', linewidth=0.5)
    plt.clabel(C, inline=True, fontsize=10)
    plt.scatter(data[:, 0], data[:, 1])