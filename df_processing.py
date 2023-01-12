import math
import os
import subprocess

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA

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


def get_one_dataframe(mri_dfs_to_join):
    dfs_to_join = [mri_dfs_to_join[cat].df for cat in mri_dfs_to_join]

    for df, cat in zip(dfs_to_join, mri_dfs_to_join):
        df['Class'] = cat

    final_df = pd.concat(dfs_to_join)
    final_df = final_df.drop(['file_name'], axis=1)

    label_enc = LabelEncoder()
    final_df['Class'] = label_enc.fit_transform(final_df['Class'])

    return final_df


def get_best_features(x, y, col, n_features='all'):
    skb = SelectKBest(mutual_info_classif, k=n_features)
    fit_res = skb.fit(x, y)

    result_df = pd.DataFrame()
    result_df['Feature_Name'] = fit_res.feature_names_in_
    result_df['Score'] = fit_res.scores_

    cols = skb.get_support(indices=True)
    chosen_features = [col[i] for i in cols]

    return result_df, chosen_features


def normalize_df(df_to_norm, columns):
    df_vals = df_to_norm.loc[:, columns].values
    norm_arr = StandardScaler().fit_transform(df_vals)

    return norm_arr


def perform_pca(normalized_array, class_seq, n_components=2):
    pca_mri = PCA(n_components=n_components)
    principal_component_mri = pca_mri.fit_transform(normalized_array)

    column_names = [f'princ_comp_{i + 1}' for i in range(n_components)]
    pca_mri_df = pd.DataFrame(data=principal_component_mri, columns=column_names)
    pca_mri_df['Class'] = class_seq.values.tolist()

    print('Explained variation per principal component: {}'.format(pca_mri.explained_variance_ratio_))

    if n_components == 2:
        sn.scatterplot(x='princ_comp_1',
                       y='princ_comp_2',
                       hue='Class',
                       data=pca_mri_df)
        plt.show()

    return pca_mri_df