import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage import feature


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

    def show(self) -> None:
        plt.imshow(self.lbp)
        plt.colorbar()
        plt.show()
