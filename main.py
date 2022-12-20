import matplotlib.pyplot as plt

import df_processing as dfp
from mri_images_df import MriDataFrame

RESOURCE_DIR_NAME = 'resources'

if __name__ == '__main__':
    plt.style.use('ggplot')

    # mri_images_dataset = ImageDataset()
    # mri_images_dataset.load_images(RESOURCE_DIR_NAME)
    # mri_images_dataset.save_dataset('processed')

    non = MriDataFrame('processed/NonDemented.csv')
    very_mild = MriDataFrame('processed/VeryMildDemented.csv')
    mild = MriDataFrame('processed/MildDemented.csv')
    moderate = MriDataFrame('processed/ModerateDemented.csv')

    class_dfs = {'NonDemented': non,
                 'VeryMildDemented': very_mild,
                 'MildDemented': mild,
                 'ModerateDemented': moderate}

    x, y = dfp.get_xy_arrays_from_df0s(class_dfs)
    best_features = dfp.get_best_features(x, y, 10, non.features[1:])

    print(f'Features chosen by chi2: {best_features[0]}\n'
          f'Features chosen by mutual_info_classif: {best_features[1]}\n'
          f'Features chosen by f_classif: {best_features[2]}')
