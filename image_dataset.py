import os

import numpy as np

import process_img as pimg
from mri_image import MriImage

from os import path


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
    def generate_histograms(self) -> None:
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

    # Call all methods with default values
    def generate_defauts(self) -> None:
        sample_image = self.image_list[-1]

        if sample_image.histogram is None:
            self.generate_histograms()

        if sample_image.fft is None:
            self.generate_ffts()

        if sample_image.lbp is None:
            self.generate_lbps(3, 24, 'uniform')

        if sample_image.glcm is None:
            self.generate_glcm([5],
                               [0, np.pi / 4, np.pi / 2, np.pi * 3 / 4])

        if sample_image.zernike:
            self.generate_zernike_moments(5)

    def save_dataset(self, path_to_dir):
        # self.generate_defauts()

        cats = self.categories

        if not path.exists(path_to_dir):
            os.makedirs(path_to_dir)

        for cat in cats:
            cat_path = f'{path_to_dir}/{cat}'

            if not path.exists(cat_path):
                os.makedirs(cat_path)

            imgs = [img for img in self.image_list if img.category == cat]

            for img in imgs:
                img_path = f'{path_to_dir}/{cat}/{img.name}'

                if not path.exists(img_path):
                    os.makedirs(img_path)

                img.save_image_with_attributes(img_path)
                return
