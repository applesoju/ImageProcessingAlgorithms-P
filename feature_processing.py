import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
from sklearn.decomposition import PCA, FastICA
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import LocallyLinearEmbedding


def get_max_acc(accs_dict):
    max_accuracy = 0.0
    best_acc_key = None

    for key in accs_dict:
        acc = accs_dict[key][0]

        if acc > max_accuracy:
            best_acc_key = key
            max_accuracy = acc

    return best_acc_key, max_accuracy


class FeatureProcessing:

    def __init__(self, df_or_csv, verbose=False):
        self.verbose = verbose

        self.feature_scores = None
        self.best_features = None

        self.norm_mean = None
        self.norm_std = None
        self.normalized_feature_df = None

        self.pca_model = None
        self.feature_pca_df = None

        self.lda_model = None
        self.feature_lda_df = None

        self.ica_model = None
        self.feature_ica_df = None

        self.lle_model = None
        self.feature_lle_df = None

        self.best_logreg_model = None
        self.best_dectree_model = None
        self.best_forest_model = None

        self.class_encoding = pd.DataFrame()

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

        self.norm_mean = fdf.mean()
        self.norm_std = fdf.std()

        self.normalized_feature_df = (fdf - self.norm_mean) / self.norm_std
        self.normalized_feature_df = self.normalized_feature_df.fillna(0)
        self.normalized_feature_df.insert(0, 'Class', self.features_df['Class'])

        return self.normalized_feature_df

    def get_best_features(self, n_features='all'):
        if self.verbose:
            print(f'Selecting {n_features} best features from a total number of {len(self.features_df.columns)}...')

        fdf = self.features_df

        label_enc = LabelEncoder()
        class_col = fdf['Class']
        class_labels = label_enc.fit_transform(class_col)

        model = SelectKBest(mutual_info_classif, k=n_features)
        x = fdf.drop(['Class'], axis=1)
        y = class_labels
        fit_res = model.fit(x, y)

        chosen_features_indices = model.get_support(indices=True).tolist()
        self.best_features = [x.columns.values.tolist()[i] for i in chosen_features_indices]

        result_df = pd.DataFrame()
        result_df['Feature_Name'] = fit_res.feature_names_in_
        result_df['Score'] = fit_res.scores_

        self.feature_scores = result_df

        if self.verbose:
            print('Best features and their scores:')

            for feat in self.best_features:
                row = result_df[result_df['Feature_Name'] == feat]
                score = np.squeeze(row['Score'].values)

                print(f'{feat}: {score}')

        return self.best_features

    def perform_pca(self, features_to_consider, n_components):
        if self.verbose:
            print('Peforming Principal Component Analysis...')

        pca = PCA(n_components=n_components)

        norm_feat_df = self.normalize_columns()

        feature_df = norm_feat_df[features_to_consider]
        feature_pca = pca.fit_transform(feature_df)

        self.feature_pca_df = pd.DataFrame(feature_pca)
        self.feature_pca_df.insert(0, 'Class', self.features_df['Class'])

        if self.verbose:
            print(f'Explained variation per principal component: {pca.explained_variance_ratio_}')

        self.pca_model = pca

        return self.feature_pca_df

    def perform_lda(self, features_to_consider, n_components):
        if self.verbose:
            print('Peforming Linear Discrimination Analysis...')

        lda = LinearDiscriminantAnalysis(n_components=n_components)

        norm_feat_df = self.normalize_columns()

        label_enc = LabelEncoder()
        class_labels = label_enc.fit_transform(self.normalized_feature_df['Class'])

        feature_df = norm_feat_df[features_to_consider]
        feature_lda = lda.fit(feature_df, class_labels).transform(feature_df)

        self.feature_lda_df = pd.DataFrame(feature_lda)
        self.feature_lda_df.insert(0, 'Class', self.features_df['Class'])

        if self.verbose:
            print(f'Explained variation: {lda.explained_variance_ratio_}')

        self.lda_model = lda

        return self.feature_lda_df

    def perform_ica(self, features_to_consider, n_components):
        if self.verbose:
            print('Peforming Independent Component Analysis...')

        ica = FastICA(n_components=n_components)

        norm_feat_df = self.normalize_columns()
        feature_df = norm_feat_df[features_to_consider]
        feature_ica = ica.fit_transform(feature_df)

        self.feature_ica_df = pd.DataFrame(feature_ica)
        self.feature_ica_df.insert(0, 'Class', self.features_df['Class'])
        self.ica_model = ica

        return self.feature_ica_df

    def perform_lle(self, features_to_consider, n_components):
        if self.verbose:
            print('Peforming Locally Linear Embedding...')

        lle = LocallyLinearEmbedding(n_components=n_components)

        norm_feat_df = self.normalize_columns()
        feature_df = norm_feat_df[features_to_consider]
        feature_lle = lle.fit_transform(feature_df)

        self.feature_lle_df = pd.DataFrame(feature_lle)
        self.feature_lle_df.insert(0, 'Class', self.features_df['Class'])
        self.lle_model = lle

        return self.feature_ica_df

    def get_best_logreg_model(self, x, y):
        if self.verbose:
            print('Looking for the most accurate Logistic Regression model...')

        accs_dict = {}

        solvers = ['lbfgs', 'newton-cg', 'sag', 'saga']
        c_vals = [0.01, 0.05, 0.1, 0.25, 0.4, 0.6, 0.8, 0.9, 0.99]

        for solver in solvers:
            for c in c_vals:
                model = LogisticRegression(multi_class='multinomial',
                                           solver=solver,
                                           C=c)
                cv = RepeatedStratifiedKFold(n_splits=8, n_repeats=100)
                n_scores = cross_val_score(model, x, y, scoring='accuracy', cv=cv, n_jobs=1)

                key = f'{solver}_{c}'
                accuracy = np.mean(n_scores)
                accs_dict[key] = [accuracy, model]

            if self.verbose:
                print(f'Finished checking models with {solver} solver.')

        best_acc_key, max_accuracy = get_max_acc(accs_dict)

        if self.verbose:
            s, c = best_acc_key.split('_')
            print(f'Found a model with the best accuracy of {max_accuracy}. Model properties:\n'
                  f'Solver: {s}\n'
                  f'C value: {c}')

        self.best_logreg_model = accs_dict[best_acc_key][1]

        return self.best_logreg_model

    def get_best_dectree_model(self, x, y):
        if self.verbose:
            print('Looking for the most accurate Decision Tree model...')

        accs_dict = {}

        crits = ['gini', 'entropy', 'log_loss']

        for crit in crits:
            model = DecisionTreeClassifier(criterion=crit)
            cv = RepeatedStratifiedKFold(n_splits=8, n_repeats=100)
            n_scores = cross_val_score(model, x, y, scoring='accuracy', cv=cv, n_jobs=1)

            accuracy = np.mean(n_scores)
            accs_dict[crit] = [accuracy, model]

            if self.verbose:
                print(f'Finished checking models with {crit} criterion.')

        best_acc_key, max_accuracy = get_max_acc(accs_dict)

        if self.verbose:
            print(f'Found a model with the best accuracy of {max_accuracy}. Model properties:\n'
                  f'Criterion: {best_acc_key}\n')

        self.best_dectree_model = accs_dict[best_acc_key][1]

        return self.best_dectree_model

    def get_best_forest_model(self, x, y):
        if self.verbose:
            print('Looking for the most accurate Random Forest model...')

        accs_dict = {}

        crits = ['gini', 'entropy', 'log_loss']

        for crit in crits:
            model = RandomForestClassifier(criterion=crit)
            cv = RepeatedStratifiedKFold(n_splits=8, n_repeats=100)
            n_scores = cross_val_score(model, x, y, scoring='accuracy', cv=cv, n_jobs=1)

            accuracy = np.mean(n_scores)
            accs_dict[crit] = [accuracy, model]

            if self.verbose:
                print(f'Finished checking models with {crit} criterion.')

        best_acc_key, max_accuracy = get_max_acc(accs_dict)

        if self.verbose:
            print(f'Found a model with the best accuracy of {max_accuracy}. Model properties:\n'
                  f'Criterion: {best_acc_key}\n')

        self.best_forest_model = accs_dict[best_acc_key][1]

        return self.best_forest_model

    def get_class_enc_and_xy(self, reduce_dims=None):
        match reduce_dims:
            case None:
                self.class_encoding['Name'] = self.features_df['Class']
                x = self.features_df.drop(['Class'], axis=1)
                x = x[self.best_features]

            case 'pca':
                self.class_encoding['Name'] = self.feature_pca_df['Class']
                x = self.feature_pca_df.drop(['Class'], axis=1)

            case 'lda':
                self.class_encoding['Name'] = self.feature_lda_df['Class']
                x = self.feature_lda_df.drop(['Class'], axis=1)

            case 'ica':
                self.class_encoding['Name'] = self.feature_ica_df['Class']
                x = self.feature_ica_df.drop(['Class'], axis=1)

            case 'lle':
                self.class_encoding['Name'] = self.feature_lle_df['Class']
                x = self.feature_lle_df.drop(['Class'], axis=1)

            case _:
                raise ValueError(f'{reduce_dims} is not a valid parameter')


        label_enc = LabelEncoder()
        self.class_encoding['Label'] = label_enc.fit_transform(self.class_encoding['Name'])

        y = self.class_encoding['Label']

        return x, y

    def multinomial_logistic_regression(self, solver=None, c_val=None, red_dim=None):
        x, y = self.get_class_enc_and_xy(reduce_dims=red_dim)

        if solver is None and c_val is None:
            model = self.get_best_logreg_model(x, y)

        else:
            model = LogisticRegression(multi_class='multinomial',
                                       solver=solver,
                                       C=c_val)
            cv = RepeatedStratifiedKFold(n_splits=8, n_repeats=10)
            n_scores = cross_val_score(model, x, y, scoring='accuracy', cv=cv, n_jobs=1)

            if self.verbose:
                print(f'Mean accuracy from Cross Validation: {np.mean(n_scores):.3f}, {np.std(n_scores):.3f}')

        print('Fitting the model...')

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=144)
        model.fit(x_train, y_train)

        score = model.score(x_test, y_test)
        if self.verbose:
            print(f'Score from prediction of test subset: {score}')

        self.best_logreg_model = model

        return self.best_logreg_model

    def desicion_tree_classifier(self, criterion=None):
        x, y = self.get_class_enc_and_xy()

        if criterion is None:
            model = self.get_best_dectree_model(x, y)

        else:
            model = DecisionTreeClassifier(criterion=criterion)
            cv = RepeatedStratifiedKFold(n_splits=8, n_repeats=10)
            n_scores = cross_val_score(model, x, y, scoring='accuracy', cv=cv, n_jobs=1)

            if self.verbose:
                print(f'Mean accuracy from Cross Validation: {np.mean(n_scores):.3f}, {np.std(n_scores):.3f}')

        print('Fitting the model...')

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=144)
        model.fit(x_train, y_train)

        score = model.score(x_test, y_test)
        if self.verbose:
            print(f'Score from prediction of test subset: {score}')

        self.best_dectree_model = model

        return self.best_dectree_model

    def random_forest_classifier(self, criterion=None):
        x, y = self.get_class_enc_and_xy()

        if criterion is None:
            model = self.get_best_forest_model(x, y)

        else:
            model = RandomForestClassifier(criterion=criterion)
            cv = RepeatedStratifiedKFold(n_splits=8, n_repeats=10)
            n_scores = cross_val_score(model, x, y, scoring='accuracy', cv=cv, n_jobs=1)

            if self.verbose:
                print(f'Mean accuracy from Cross Validation: {np.mean(n_scores):.3f}, {np.std(n_scores):.3f}')

        print('Fitting the model...')

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=144)
        model.fit(x_train, y_train)

        score = model.score(x_test, y_test)
        if self.verbose:
            print(f'Score from prediction of test subset: {score}')

        self.best_forest_model = model

        return self.best_forest_model

    def compare_models(self):
        raise NotImplementedError

    def prob_predict(self, features, features_to_consider, model):
        norm_features = (features - self.norm_mean) / self.norm_std
        norm_features = norm_features.fillna(0)
        only_features = norm_features.drop(['Class'], axis=1)

        class_names = self.class_encoding['Name'].unique()
        class_labels = self.class_encoding['Label'].unique()
        class_enc = [0 for _ in class_labels]

        j = 0
        for i in class_labels:
            class_enc[i] = class_names[j]
            j += 1

        # pca = self.pca_model

        best_features = only_features[features_to_consider]
        # pca_only_feats = pca.transform(best_features)

        match model:
            case 'logreg':
                pred = self.best_logreg_model.predict_proba(best_features).reshape(-1, 1)

            case 'dectree':
                pred = self.best_dectree_model.predict_proba(best_features).reshape(-1, 1)

            case 'forest':
                pred = self.best_forest_model.predict_proba(best_features).reshape(-1, 1)

            case _:
                raise ValueError(f'{model} is not a recognizable model.')

        if self.verbose:
            print(f'Predicted probabilities:')

            for i in range(len(class_enc)):
                print(f'{class_enc[i]}: {pred[i]}')

            print(f"Correct class: {np.squeeze(features['Class'].values)}")

        return pred