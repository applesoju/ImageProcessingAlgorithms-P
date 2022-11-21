import random

import cv2
import matplotlib.pyplot as plt

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

    # Show a random image from the dataset
    def show_random_image(self) -> None:
        random_index = random.randint(0, len(self.image_list))
        random_img = self.image_list[random_index]
        random_img.show_image()

    # Show an image from every category
    # def show_image_from_all_cat(self, scale) -> None:

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

    # Show a histogram of an image from every category
    # def show_histogram_from_all_cat(self, scale) -> None:
    #     img_from_cat = []
    #     cat_count = len(self.categories)
    #
    #     for cat in self.categories:
    #         img_from_cat = [img for img in self.image_list if img.category == cat]
    #
    #     random_indexes = [random.randint(0, len(img_from_cat[i])) for i in self.categories]
    #     images = [img_from_cat[i][idx] for idx, i in (random_indexes, self.categories)]
    #
    #     layout = (cat_count // 2, 2) if cat_count % 2 == 0 else (cat_count, 1)
    #     plt.subplots(layout[0], layout[1], figsize=(16, 9))
    #
    #     for i, img in enumerate(images):
    #         hist = cv2.calcHist([img.image],
    #                             [0],
    #                             None,
    #                             [256],
    #                             [0, 256])
    #         plt.subplot(layout[0] * 100 +
    #                     layout[1] * 10 + i + 1)
    #         plt.plot(hist, color='black')
    #         plt.xlabel('Pixel value')
    #         plt.ylabel('Pixel count')
    #         plt.title(f'Name = {img.name}, Category = {img.category}')
    #         plt.grid()
    #         plt.yscale(scale)
    #
    #     plt.show()
