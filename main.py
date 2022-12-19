import matplotlib.pyplot as plt
import numpy as np
from mri_images_df import MriDataFrame

from image_dataset import ImageDataset

import os

RESOURCE_DIR_NAME = 'resources'

if __name__ == '__main__':
    plt.style.use('ggplot')

    # mri_images_dataset = ImageDataset()
    # mri_images_dataset.load_images(RESOURCE_DIR_NAME)
    # mri_images_dataset.save_dataset('processed')

    mdf = MriDataFrame('processed/ModerateDemented.csv')
    # mdf.print()

    print(len(mdf.features))