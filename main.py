import matplotlib.pyplot as plt
import numpy as np
from mri_images_df import MriDataFrame
import df_processing as dfp
from image_dataset import ImageDataset
import seaborn as sn

import os

RESOURCE_DIR_NAME = 'resources'

if __name__ == '__main__':
    plt.style.use('ggplot')

    # mri_images_dataset = ImageDataset()
    # mri_images_dataset.load_images(RESOURCE_DIR_NAME)
    # mri_images_dataset.save_dataset('processed')

    non = MriDataFrame('processed/NonDemented.csv')
    very_mild = MriDataFrame('processed/VeryMildDemented.csv')
    mild = MriDataFrame('processed/MildDemented.csv')
    moderate = MriDataFrame('processed/ModerateDemented.csv')

    class_dfs = {'Non': non,
                 'Very_Mild': very_mild,
                 'Mild': mild,
                 'Moderate': moderate}

    # dfp.plot_distributions_of_feature(class_dfs, 1)
    # dfp.save_dist_of_all_features(class_dfs, 'feature_distributions')

