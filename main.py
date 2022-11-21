from mri_image import MriImage
import matplotlib.pyplot as plt
import numpy as np
import cv2
import process_img as pimg
import sys

RESOURCE_DIR_NAME = 'resources'

if __name__ == '__main__':
    file_list = pimg.get_images_path_list(RESOURCE_DIR_NAME)
    images = []

    for file_path in file_list:
        _, category, name = file_path.split('/')
        new_image = MriImage(name, category, file_path)
        images.append(new_image)

    images[0].show_image()
