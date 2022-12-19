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

    moderate = MriDataFrame('processed/ModerateDemented.csv')
    # moderate.print()
    fig, axs = plt.subplots()

    moderate.plot(axs, 1)
    moderate.plot(axs, 2)
    
    plt.show()