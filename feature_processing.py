import math
import os
import subprocess

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

import image_dataset_processing


class FeatureProcessing:

    def __init__(self, df_or_csv):
        if type(df_or_csv) is pd.DataFrame:
            self.features_df = df_or_csv

        elif type(df_or_csv) is str:
            if not os.path.exists(df_or_csv):
                raise FileNotFoundError

            self.features_df = pd.read_csv(df_or_csv)

        else:
            raise TypeError

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


def perform_pca(normalized_array, class_cols, n_components=2):
    pca_mri = PCA(n_components=n_components)
    principal_component_mri = pca_mri.fit_transform(normalized_array)

    column_names = [f'princ_comp_{i + 1}' for i in range(n_components)]
    pca_mri_df = pd.DataFrame(data=principal_component_mri, columns=column_names)
    pca_mri_df['Class_Label'] = class_cols['Class_Label'].values.tolist()
    pca_mri_df['Class_Name'] = class_cols['Class_Name'].values.tolist()

    print('Explained variation per principal component: {}'.format(pca_mri.explained_variance_ratio_))

    if n_components == 2:
        sn.scatterplot(x='princ_comp_1',
                       y='princ_comp_2',
                       hue='Class_Name',
                       data=pca_mri_df)
        plt.show()

    return pca_mri_df


def multinomial_logistic_regression_cv(data_df):
    x = data_df.drop(['Class_Label', 'Class_Name'], axis=1)
    y = data_df['Class_Label']

    model = LogisticRegression(multi_class='multinomial', solver='lbfgs')
    cv = RepeatedStratifiedKFold(n_splits=8, n_repeats=4, random_state=144)
    n_scores = cross_val_score(model, x, y, scoring='accuracy', cv=cv, n_jobs=1)

    print(f'Mean accuracy from Cross Validation: {np.mean(n_scores):.3f}, {np.std(n_scores):.3f}')

    return n_scores


def multinomial_logistic_regression_random_predict(data_df, class_enc):
    x = data_df.drop(['Class_Label', 'Class_Name'], axis=1)
    y = data_df['Class_Label']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=144)

    model = LogisticRegression(multi_class='multinomial', solver='lbfgs')
    model.fit(x_train, y_train)

    random_idx = np.random.randint(len(x_test))
    random_sample = x_test.iloc[random_idx, :]

    pred = model.predict_proba([random_sample]).reshape(-1, 1)

    print(f'Predicted probabilities:')
    for i in range(len(class_enc)):
        print(f'{class_enc[i]}: {pred[i]}')

    print(f'Correct class: {class_enc[y_test.iloc[random_idx]]}')

    return pred
