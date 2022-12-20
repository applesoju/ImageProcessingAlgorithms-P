import matplotlib.pyplot as plt
import numpy as np
from mri_images_df import MriDataFrame
import df_processing as dfp
from image_dataset import ImageDataset
import seaborn as sn
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import f_classif

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

    class_dfs = {'NonDemented': non,
                 'VeryMildDemented': very_mild,
                 'MildDemented': mild,
                 'ModerateDemented': moderate}

    x, y = dfp.get_xy_arrays_from_dfs(class_dfs)
    x_chi = SelectKBest(chi2, k=10).fit_transform(x, y)
    x_mic = SelectKBest(mutual_info_classif, k=10).fit_transform(x, y)
    x_fc = SelectKBest(f_classif, k=10).fit_transform(x, y)

    idx_list = []
    for xn in x_chi[0]:
        idx = x[0].tolist().index(xn)
        idx_list.append(idx)
    print(idx_list)

    idx_list = []
    for xn in x_mic[0]:
        idx = x[0].tolist().index(xn)
        idx_list.append(idx)
    print(idx_list)

    idx_list = []
    for xn in x_fc[0]:
        idx = x[0].tolist().index(xn)
        idx_list.append(idx)
    print(idx_list)
