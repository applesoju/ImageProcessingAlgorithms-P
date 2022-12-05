import matplotlib.pyplot as plt
from skimage import feature


class LBP:
    def __init__(self, image, radius, n_points, method) -> None:
        self.radius = radius
        self.n_points = n_points
        self.method = method

        self.lbp = feature.local_binary_pattern(image, self.n_points, self.radius, method=self.method)
        self.hist = self.lbp.ravel()

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
