from feature_processing import FeatureProcessing
from image_dataset_processing import ImageDatasetProcessing
from image_dataset_processing import LBP_PARAMS, ZM_PARAMS, GLCM_PARAMS
import cv2

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

    # Use this variable if features were already determined
    # filepath = 'feature_processing/features_df.csv'

    # Create class and load a DataFrame containing features
    feature_proc = FeatureProcessing(filepath, verbose=True)

    # Determine a given number of best features
    best_features = feature_proc.get_best_features(30)
    print('---------------------------------------------------------------')

    # Perform Principal Component Analysis
    pca = feature_proc.perform_pca(best_features, n_components=3)
    print('---------------------------------------------------------------')

    # Perform Multinomial Logistic Regression
    logreg_model = feature_proc.multinomial_logistic_regression(solver='newton-cg', c_val=0.75)
    print('---------------------------------------------------------------')

    idp = ImageDatasetProcessing()

    # Prediction
    image_path = 'resources/test/MildDemented/26.jpg'
    features = idp.process_image(image_path)
    prediction = feature_proc.prob_predict(features, best_features)
