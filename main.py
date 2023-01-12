import matplotlib.pyplot as plt
import pandas as pd

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

    # x, y = dfp.get_xy_arrays_from_dfs(class_dfs)
    final_df = dfp.get_one_dataframe(class_dfs)

    x = final_df.drop(['Class'], axis=1)
    y = final_df['Class']

    bf_df, best_features = dfp.get_best_features(x, y, non.features[1:], 20)

    print('Features chosen by mutual_info_classif:')
    for feat in best_features:
        print(feat)

    norm_arr_mri = dfp.normalize_df(final_df, best_features)
    norm_df = pd.DataFrame(norm_arr_mri, columns=best_features)

    pca_df = dfp.perform_pca(norm_arr_mri, final_df['Class'], 3)


    # środowisko orange
    # pca - 3 cechy