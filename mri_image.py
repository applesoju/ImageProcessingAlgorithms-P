import cv2
import mahotas
import matplotlib.pyplot as plt
import numpy as np

from lbp import LBP
from skimage import feature
import pandas as pd


class MriImage:
    # Represents an MRI image

    def __init__(self, name, category, file_path) -> None:
        self.histogram = None
        self.fft = None

        self.lbp = None
        self.glcm = None
        self.zernike = None

        self.name = name
        self.category = category
        self.image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

    # Show an MRI image
    def show_image(self) -> None:
        # Generate a label for the image
        label = f'Name = {self.name}, ' \
                f'Category = {self.category}'

        # Display the image
        cv2.imshow(label, self.image)
        cv2.waitKey(0)

    # Creates a histogram of the image
    def create_histogram(self) -> None:
        self.histogram = cv2.calcHist(
            [self.image],  # source image
            [0],  # channel [0] is grayscale
            None,  # mask
            [256],  # size of the histogram
            [0, 256]  # range of the histogram
        )

    # Displays the histogram of the image
    def show_histogram(self, scale) -> None:
        if self.histogram is None:
            self.create_histogram()

        plt.plot(self.histogram, color='black')
        plt.xlabel('Pixel value')
        plt.ylabel('Pixel count')
        plt.title(f'Name = {self.name}, Category = {self.category}')
        plt.grid()
        plt.yscale(scale)
        plt.show()

    # Creates a Fourier transform of the image
    def create_fft(self) -> None:
        fourier = np.fft.fft2(self.image)
        fshift = np.fft.fftshift(fourier)
        self.fft = 20 * np.log(np.abs(fshift))

    # Displays the Fourier Transform of the image
    def show_fft(self) -> None:
        plt.imshow(self.fft)
        plt.show()

    # Creates a Local Binary Pattern descriptor of the image
    def create_lbp(self, radius, n_points, method) -> None:
        self.lbp = LBP(self.image, radius, n_points, method)

    # Displays a Local Binary Pattern descriptor as an image
    def show_lbp_image(self) -> None:
        self.lbp.show()

    # Displays a Local Binary Pattern descriptor as a histogram
    def show_lbp_hist(self) -> None:
        self.lbp.show_hist()

    # Creates a Zernike Moments of the image
    def create_zernike_moments(self, radius):
        self.zernike = mahotas.features.zernike_moments(self.image, radius)

    def calculate_glcm(self, dist, angles, levels, sym, norm):
        props = ['dissimilarity', 'correlation', 'homogeneity', 'contrast', 'ASM', 'energy']
        angles_in_deg = np.degrees(angles)
        columns = [
            f'{prop}_{int(angle)}' for prop in props for angle in angles_in_deg
        ]

        glcm = feature.graycomatrix(
            self.image,
            dist,
            angles,
            levels,
            sym,
            norm
        )
        glcm_props = [propery for name in props for propery in feature.graycoprops(glcm, name)[0]]
        self.glcm = pd.DataFrame([glcm_props], columns=columns)

    def save_image_with_attributes(self, path) -> None:
        cv2.imwrite(f'{path}/orig_{self.name}', self.image)
        cv2.imwrite(f'{path}/fft_{self.name}', self.fft)
        cv2.imwrite(f'{path}/lbp-image_{self.name}', self.lbp.lbp)

        n_bins = int(self.lbp.lbp.max() + 1)
        plt.hist(self.lbp.hist,
                 bins=n_bins,
                 range=(0, n_bins),
                 edgecolor='black')
        plt.savefig(f'{path}/lbp-hist_{self.name}')
        plt.clf()

        plt.plot(self.histogram, color='black')
        plt.savefig(f'{path}/hist_{self.name}')
        plt.clf()

        zernike_name = f'{path}/zernike_{self.name}'[:-4] + '.txt'
        with open(zernike_name, 'w') as output:
            for i in self.zernike:
                output.write(f'{i}\n')

        glcm_name = f'{path}/glcm_{self.name}'[:-4] + '.csv'
        self.glcm.to_csv(glcm_name)
