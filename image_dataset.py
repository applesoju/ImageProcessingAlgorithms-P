import random

import cv2

import process_img as pimg
from mri_image import MriImage


class ImageDataset:
    def __init__(self) -> None:
        self.image_list = []

    def load_images(self, dir_path) -> None:
        file_path_list = pimg.get_images_path_list(dir_path)

        for file_path in file_path_list:
            _, category, name = file_path.split('/')
            new_image = MriImage(name, category, file_path)
            self.image_list.append(new_image)

    def show_random_image(self):
        random_index = random.randint(0, len(self.image_list))
        random_img = self.image_list[random_index]
        random_img.show_image()
