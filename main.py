from image_dataset_processing import ImageDatasetProcessing
from image_dataset_processing import LBP_PARAMS, ZM_PARAMS, GLCM_PARAMS
import numpy as np
from feature_processing import FeatureProcessing

RESOURCE_DIR_NAME = 'resources/test'
PROCESSED_DIR_NAME = 'feature_processing'
FEATURES_FILE_NAME = 'features_df'

if __name__ == '__main__':
    # Create class and load images by class from a given dir
    dataset_proc = ImageDatasetProcessing(RESOURCE_DIR_NAME, verbose=True)
    print('---------------------------------------------------------------')

    # Generate LBP, Zernike Moments and Gray Level Co-occurance Matrix with given parameters for every image
    dataset_proc.generate_features_for_dataset(LBP_PARAMS,
                                               ZM_PARAMS,
                                               GLCM_PARAMS)
    print('---------------------------------------------------------------')

    # Save a DataFrame containing all features to a csv file
    filepath = dataset_proc.save_features_to_csv(PROCESSED_DIR_NAME,
                                                 FEATURES_FILE_NAME)
    print('---------------------------------------------------------------')

    # filepath = 'feature_processing/features_df.csv'

    # Create class and load a DataFrame containing features
    feature_proc = FeatureProcessing(filepath, verbose=True)

    # Determine a given number of best features
    best_features = feature_proc.get_best_features(20)
    print('---------------------------------------------------------------')





    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
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
