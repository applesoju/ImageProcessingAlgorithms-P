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
            plt.xlabel('Pixel value')
            plt.ylabel('Pixel count')
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

    def show_hist_from_each_cat(self, scale) -> None:
        cat_count = len(self.categories)
        layout = (cat_count // 2, 2) if cat_count % 2 == 0 else (cat_count, 1)

        images_by_categories = [
            [img for img in self.image_list if img.category == category] for category in self.categories
        ]

        random_indexes = [random.randint(0, len(cat)) for cat in images_by_categories]
        for i, idx in enumerate(random_indexes):
            img = images_by_categories[i][idx]
            hist = cv2.calcHist([img.image],
                                [0],
                                None,
                                [256],
                                [0, 256])

            plt.subplot(layout[0], layout[1], i + 1)
            plt.plot(hist, color='black')
            plt.xlabel('Pixel value')
            plt.ylabel('Pixel count')
            plt.title(f'Name = {img.name}, Category = {img.category}')
            plt.grid()
            plt.yscale(scale)

        plt.show()

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

    def show_mean_histograms(self, scale):
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
            plt.yscale(scale)

        plt.show()

