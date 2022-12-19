import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn
import os
import subprocess


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
        figure_path = f'{dir_path}/{f}.png'
        save_distributions_of_feature(class_dict, f, figure_path)
