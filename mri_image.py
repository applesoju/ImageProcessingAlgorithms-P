import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage import feature
from lbp import LBP
import mahotas


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