import math

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn
import os
import subprocess
import numpy as np

import image_dataset


def save_distributions_of_feature(class_dict, feature, fig_path):
    class_feature_df = pd.DataFrame()

    for cd in class_dict:
        mdf = class_dict[cd]

        class_feature_df[cd] = mdf.df[feature]

    sn.displot(class_feature_df, kind='kde', common_norm=False)
    plt.savefig(fig_path)
    plt.close()


def save_dist_of_all_features(class_dict, dir_path):
    if not os.path.exists(dir_path):
        subprocess.call(['mkdir', dir_path], shell=True)

    features = next(iter(class_dict.values())).features

    for f in features[1:]:
        fp = image_dataset.FEATURES_PARAMS

        dir_for_figs = f"{fp['lbp'][0]}-{fp['lbp'][1]}-{fp['lbp'][2]}_" \
                       f"{fp['zm'][0]}_" \
                       f"{int(fp['glcm'][0][0])}-" \
                       f"{int(math.degrees(fp['glcm'][1][0]))}-{int(math.degrees(fp['glcm'][1][1]))}-" \
                       f"{int(math.degrees(fp['glcm'][1][2]))}-{int(math.degrees(fp['glcm'][1][3]))}"

        fig_dir_path = f'{dir_path}\\{dir_for_figs}'
        if not os.path.exists(fig_dir_path):
            subprocess.call(['mkdir', fig_dir_path], shell=True)

        figure_path = f'{fig_dir_path}/{f}.png'
        save_distributions_of_feature(class_dict, f, figure_path)


def get_xy_arrays_from_dfs(dfs):
    feature_list = []
    class_list = []

    for key in dfs:
        features, classes = dfs[key].get_as_list(list(dfs.keys()))

        feature_list += features
        class_list += classes

    return np.array(feature_list), np.array(class_list)

def get_best_features(x, y, n_features):
    raise NotImplementedError