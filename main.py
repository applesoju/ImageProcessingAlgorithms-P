from feature_processing import FeatureProcessing
from image_dataset_processing import ImageDatasetProcessing
from image_dataset_processing import LBP_PARAMS, ZM_PARAMS, GLCM_PARAMS
import cv2

RESOURCE_DIR_NAME = 'resources/augmented'
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
    best_features = feature_proc.get_best_features(20)
    print('---------------------------------------------------------------')

    # Perform Principal Component Analysis
    pca = feature_proc.perform_pca(best_features, n_components=3)

    # Perform Linear Discrimination Analysis
    lda = feature_proc.perform_lda(best_features, n_components=3)

    # Perform Linear Discrimination Analysis
    ica = feature_proc.perform_ica(best_features, n_components=3)

    # Perform Linear Discrimination Analysis
    lle = feature_proc.perform_lle(best_features, n_components=3)

    # print('---------------------------------------------------------------')

    # Get model using Multinomial Logistic Regression
    # logreg_model = feature_proc.multinomial_logistic_regression(solver='saga', c_val=0.9, red_dim='lle')

    # Get model using Decision Tree Clasificator
    # dectree_model = feature_proc.desicion_tree_classifier(criterion='entropy')

    # Get model using Random Forest Clasificator
    # forest_model = feature_proc.random_forest_classifier(criterion='log_loss', n_est=10)

    # Check all combinations of models in search of the best one
    best_model = feature_proc.find_best_model()

    print('---------------------------------------------------------------')

    idp = ImageDatasetProcessing()

    # Prediction
    image_paths = ['resources/augmented/MildDemented/799dd911-7abd-4dc6-b2f3-704a4b4eeee8.jpg',
                   'resources/augmented/ModerateDemented/0585089e-4248-4686-980c-68bb722e048d.jpg',
                   'resources/augmented/NonDemented/8172e7dc-7b83-4efe-bf40-9bc6ee2a0598.jpg',
                   'resources/augmented/VeryMildDemented/49350611-e865-4f7b-8f14-c3e9f2eb8177.jpg']

    for image_path in image_paths:
        features = idp.process_image(image_path)
        prediction = feature_proc.prob_predict(features, best_features, 'forest')
        print('---------------------------------------------------------------')
