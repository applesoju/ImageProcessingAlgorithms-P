import matplotlib.pyplot as plt
import numpy as np
from skimage import feature
import pandas as pd


class LBP:
    def __init__(self, image, radius, n_points, method, index) -> None:
        self.radius = radius
        self.n_points = n_points
        self.method = method

        self.lbp = feature.local_binary_pattern(image, self.n_points, self.radius, method=self.method)

        self.n_vals = int(self.lbp.max() + 1)
        hist = np.zeros(self.n_vals)

        for val in range(self.n_vals):
            hist[val] = np.count_nonzero(self.lbp == val) / self.lbp.size

        column_names = [f'lbp_{i:03d}' for i in range(self.n_vals)]
        self.df = pd.DataFrame(data=[hist],
                               index=[index],
                               columns=column_names)

    def show_hist(self) -> None:
        n_bins = int(self.lbp.max() + 1)
        plt.hist(self.hist,
                 bins=n_bins,
                 range=(0, n_bins),
                 edgecolor='black')
        plt.show()

    def show(self) -> None:
        plt.imshow(self.lbp)
        plt.colorbar()
        plt.show()
