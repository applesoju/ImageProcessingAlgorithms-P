import matplotlib.pyplot as plt
import numpy as np

from image_dataset import ImageDataset

import os

RESOURCE_DIR_NAME = 'resources'

if __name__ == '__main__':
    plt.style.use('ggplot')

    mri_images_dataset = ImageDataset()

    mri_images_dataset.load_images(RESOURCE_DIR_NAME)
    mri_images_dataset.save_dataset('processed')

    # mri_images_dataset.save_dataset('processed')
    # mri_images_dataset.generate_lbps(3, 24, 'uniform')
    # mri_images_dataset.generate_zernike_moments(5)
    # mri_images_dataset.generate_glcm([5], [0, np.pi / 4, np.pi / 2, np.pi * 3 / 4])
