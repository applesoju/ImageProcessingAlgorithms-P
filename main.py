from image_dataset_processing import ImageDatasetProcessing
import numpy as np

RESOURCE_DIR_NAME = 'resources'
PROCESSED_DIR_NAME = 'feature_processing'
FEATURES_FILE_NAME = 'features_df'

if __name__ == '__main__':
    dataset_proc = ImageDatasetProcessing(RESOURCE_DIR_NAME)
    dataset_proc.save_features_to_csv(PROCESSED_DIR_NAME,
                                      FEATURES_FILE_NAME,
                                      verbose=True)

    # mri_images_dataset = ImageDataset()
    # mri_images_dataset.load_images(RESOURCE_DIR_NAME)
    # mri_images_dataset.save_dataset('processed')

    # non = MriDataFrame('processed/NonDemented.csv')
    # very_mild = MriDataFrame('processed/VeryMildDemented.csv')
    # mild = MriDataFrame('processed/MildDemented.csv')
    # moderate = MriDataFrame('processed/ModerateDemented.csv')
    #
    # class_dfs = {'NonDemented': non,
    #              'VeryMildDemented': very_mild,
    #              'MildDemented': mild,
    #              'ModerateDemented': moderate}
    #
    # # x, y = dfp.get_xy_arrays_from_dfs(class_dfs)
    # final_df, class_encoding = dfp.get_one_dataframe(class_dfs)
    #
    # x = final_df.drop(['Class'], axis=1)
    # y = final_df['Class']
    #
    # bf_df, best_features = dfp.get_best_features(x, y, non.features[1:], 20)
    #
    # print('Features chosen by mutual_info_classif:')
    # for feat in best_features:
    #     print(feat)
    #
    # norm_arr_mri = dfp.normalize_df(final_df, best_features)
    # norm_df = pd.DataFrame(norm_arr_mri, columns=best_features)
    #
    # class_seq = final_df['Class'].map({i: class_encoding[i] for i in range(len(class_encoding))})
    #
    # class_cols = pd.DataFrame()
    # class_cols['Class_Name'] = class_seq
    # class_cols['Class_Label'] = final_df['Class']
    #
    # pca_df = dfp.perform_pca(norm_arr_mri, class_cols, 3)
    #
    # scores = dfp.multinomial_logistic_regression_cv(pca_df)
    # probs = dfp.multinomial_logistic_regression_random_predict(pca_df, class_encoding)
