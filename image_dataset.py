import random

import cv2
import matplotlib.pyplot as plt
import numpy as np

import process_img as pimg
from mri_image import MriImage


class ImageDataset:
    # Represents a dataset that contains some images

    def __init__(self) -> None:
        self.image_list = []
        self.categories = []
        self.mean_histograms = []
        self.fft_from_mean_hist = []

    # Load images from given path to the dataset
    def load_images(self, dir_path) -> None:
        file_path_list = pimg.get_images_path_list(dir_path)

        for file_path in file_path_list:
            _, category, name = file_path.split('/')

            if category not in self.categories:
                self.categories.append(category)

            new_image = MriImage(name, category, file_path)
            self.image_list.append(new_image)

    # Show a random image from the dataset
    def show_random_image(self) -> None:
        random_index = random.randint(0, len(self.image_list))
        random_img = self.image_list[random_index]
        random_img.show_image()

    # Show an image from every category
    def show_image_from_each_cat(self) -> None:
        cat_count = len(self.categories)
        layout = (cat_count // 2, 2) if cat_count % 2 == 0 else (cat_count, 1)

        images_by_categories = [
            [img for img in self.image_list if img.category == category] for category in self.categories
        ]

        random_indexes = [random.randint(0, len(cat)) for cat in images_by_categories]
        for i, idx in enumerate(random_indexes):
            img = images_by_categories[i][idx]

            plt.subplot(layout[0], layout[1], i + 1)
            plt.imshow(img.image)
            plt.title(f'Name = {img.name}, Category = {img.category}')

        plt.show()

    # Show a random image from the specified category
    def show_image_from_category(self, category) -> None:
        images_from_cat = [img for img in self.image_list if img.category == category]

        if not images_from_cat:
            print(f'Error - Category "{category}" does not exist.')
            return

        random_index = random.randint(0, len(images_from_cat))
        random_img_from_cat = images_from_cat[random_index]
        random_img_from_cat.show_image()

    # Show a histogram of a random image
    def show_random_histogram(self, scale) -> None:
        random_index = random.randint(0, len(self.image_list))
        random_img = self.image_list[random_index]
        random_img.show_histogram(scale)

    # Shows histograms of random images from each category
    def show_hist_from_each_cat(self, scale) -> None:
        cat_count = len(self.categories)
        layout = (cat_count // 2, 2) if cat_count % 2 == 0 else (cat_count, 1)

        images_by_categories = [
            [img for img in self.image_list if img.category == category] for category in self.categories
        ]

        random_indexes = [random.randint(0, len(cat)) for cat in images_by_categories]
        for i, idx in enumerate(random_indexes):
            random_img = images_by_categories[i][idx]

            if random_img.histogram is None:
                random_img.create_histogram()

            hist = random_img.histogram

            plt.subplot(layout[0], layout[1], i + 1)
            plt.plot(hist, color='black')
            plt.xlabel('Pixel value')
            plt.ylabel('Pixel count')
            plt.title(f'Name = {random_img.name}, Category = {random_img.category}')
            plt.grid()
            plt.yscale(scale)

        plt.show()

    # Computes mean histograms of images from each category
    def compute_mean_hist(self) -> None:
        mean_histograms = []

        images_by_categories = [
            [img for img in self.image_list if img.category == category] for category in self.categories
        ]

        for idx, _ in enumerate(self.categories):
            sample_img = images_by_categories[idx][0]
            if not sample_img.histogram:
                sample_img.create_histogram()

            img_shape = images_by_categories[idx][0].histogram.shape
            mean_image = np.zeros(shape=img_shape)

            for img in images_by_categories[idx]:
                if img.histogram is None:
                    img.create_histogram()

                mean_image = np.add(img.histogram, mean_image)

            mean_image = mean_image / len(images_by_categories[idx])
            mean_histograms.append(mean_image)

        self.mean_histograms = mean_histograms

    # Shows mean histograms of images from each category
    def show_mean_histograms(self) -> None:
        cat_count = len(self.categories)
        layout = (cat_count // 2, 2) if cat_count % 2 == 0 else (cat_count, 1)

        if not self.mean_histograms:
            self.compute_mean_hist()

        for i, mean_hist in enumerate(self.mean_histograms):

            plt.subplot(layout[0], layout[1], i + 1)
            plt.plot(mean_hist, color='black')
            plt.xlabel('Pixel value')
            plt.ylabel('Pixel count')
            plt.title(f'Mean Histogram, Category = {self.categories[i]}')
            plt.grid()
            plt.ylim([0, 175])
            # plt.yscale(scale)

        plt.show()

    # Shows Fourier Transform of random images from each category
    def show_fft_from_each_cat(self) -> None:
        cat_count = len(self.categories)
        layout = (cat_count // 2, 2) if cat_count % 2 == 0 else (cat_count, 1)

        images_by_categories = [
            [img for img in self.image_list if img.category == category] for category in self.categories
        ]

        random_indexes = [random.randint(0, len(cat)) for cat in images_by_categories]
        for i, idx in enumerate(random_indexes):
            random_img = images_by_categories[i][idx]

            if random_img.fft is None:
                random_img.create_fft()

            fft = images_by_categories[i][idx].fft

            plt.subplot(layout[0], layout[1], i + 1)
            plt.imshow(fft)
            plt.title(f'Name = {random_img.name}, Category = {random_img.category}')

        plt.show()

    # Compute Fourier Transforms of mean histograms
    def compute_fft_from_mean_hist(self) -> None:
        generated_ffts = []

        if not self.mean_histograms:
            self.compute_mean_hist()

        for mh in self.mean_histograms:
            fourier = np.fft.fft(mh)
            generated_ffts.append(fourier)

        self.fft_from_mean_hist = generated_ffts

    # Generate LBP of all images using given parameters
    def generate_lbps(self, radius, n_points, method) -> None:
        for img in self.image_list:
            img.create_lbp(radius, n_points, method)

    # Generate Zernike Moments of all images
    def generate_zernike_moments(self, radius) -> None:
        for img in self.image_list:
            img.create_zernike_moments(radius)
