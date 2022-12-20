import math

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn
import os
import subprocess
import numpy as np

import image_dataset

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import f_classif


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

def get_best_features(x, y, n_features, feature_list):
    x_chi = SelectKBest(chi2, k=n_features).fit_transform(x, y)
    x_mic = SelectKBest(mutual_info_classif, k=n_features).fit_transform(x, y)
    x_fc = SelectKBest(f_classif, k=n_features).fit_transform(x, y)

    chosen_features = [[], [], []]

    for i in range(n_features):
        chi_idx = x[0].tolist().index(x_chi[0][i])
        chosen_features[0].append(feature_list[chi_idx])

        mic_idx = x[0].tolist().index(x_mic[0][i])
        chosen_features[1].append(feature_list[mic_idx])

        fc_idx = x[0].tolist().index(x_fc[0][i])
        chosen_features[2].append(feature_list[fc_idx])

    return chosen_features