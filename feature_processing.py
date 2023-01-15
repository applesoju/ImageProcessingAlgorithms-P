import os

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


class FeatureProcessing:

    def __init__(self, df_or_csv, verbose=False):
        self.verbose = verbose
        self.normalized_feature_df = None
        self.feature_scores = None

        if type(df_or_csv) is pd.DataFrame:
            self.features_df = df_or_csv

        elif type(df_or_csv) is str:
            if not os.path.exists(df_or_csv):
                raise FileNotFoundError

            self.features_df = pd.read_csv(df_or_csv, index_col=0)

        else:
            raise TypeError

    def normalize_columns(self):
        fdf = self.features_df.drop(['Class'], axis=1)
        self.normalized_feature_df = (fdf - fdf.mean()) / fdf.std()
        self.normalized_feature_df.insert(0, 'Class', self.features_df['Class'])

        return self.normalized_feature_df

    def get_best_features(self, n_features='all'):
        if self.verbose:
            print(f'Selecting {n_features} best features from a total number of {len(self.features_df.columns)}...')

        norm_fdf = self.normalized_feature_df
        if norm_fdf is None:
            norm_fdf = self.normalize_columns()

        label_enc = LabelEncoder()
        class_col = norm_fdf['Class']
        class_labels = label_enc.fit_transform(class_col)

        model = SelectKBest(mutual_info_classif, k=n_features)
        x = norm_fdf.drop(['Class'], axis=1)
        y = class_labels
        fit_res = model.fit(x, y)

        chosen_features_indices = model.get_support(indices=True).tolist()
        chosen_features = [x.columns.values.tolist()[i] for i in chosen_features_indices]

        result_df = pd.DataFrame()
        result_df['Feature_Name'] = fit_res.feature_names_in_
        result_df['Score'] = fit_res.scores_
        result_df = result_df.sort_values(by=['Score'], ascending=False)

        self.feature_scores = result_df

        if self.verbose:
            print('Best features and their scores:')

            for feat in chosen_features:
                row = result_df[result_df['Feature_Name'] == feat]
                score = np.squeeze(row['Score'].values)

                print(f'{feat}: {score}')

        return chosen_features


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
