import random

import matplotlib.pyplot as plt
import numpy as np

import process_img as pimg
from mri_image import MriImage


class ImageDataset:
    # Represents a dataset that contains some images

    def __init__(self) -> None:
        self.image_list = []
        self.categories = []

    # Load images from given path to the dataset
    def load_images(self, dir_path) -> None:
        file_path_list = pimg.get_images_path_list(dir_path)

        for file_path in file_path_list:
            _, category, name = file_path.split('/')

            if category not in self.categories:
                self.categories.append(category)

            new_image = MriImage(name, category, file_path)
            self.image_list.append(new_image)

    # Create histograms of all images
    def create_histograms(self) -> None:
        for img in self.image_list:
            img.create_histogram()

    # Generate Fourier transforms for all images
    def generate_ffts(self) -> None:
        for img in self.image_list:
            img.create_fft()

    # Generate LBP of all images using given parameters
    def generate_lbps(self, radius, n_points, method) -> None:
        for img in self.image_list:
            img.create_lbp(radius, n_points, method)

    # Generate Zernike Moments of all images
    def generate_zernike_moments(self, radius) -> None:
        for img in self.image_list:
            img.create_zernike_moments(radius)

    # Generate Gray Level Co-occurance Matrices for all images
    def generate_glcm(self, distances, angles, levels=256, symmetric=True, normed=True) -> None:
        for img in self.image_list:
            print(img.name)
            img.calculate_glcm(distances, angles, levels, symmetric, normed)
