import os
import subprocess

import numpy as np
import pandas as pd

from mri_image import MriImage

LBP_PARAMS = [[1, 2, 3], 8, 'uniform']
ZM_PARAMS = [[1, 2, 3, 4]]
GLCM_PARAMS = [[1, 2, 3],
               [0, np.pi / 12, np.pi / 6, np.pi / 4,  # DEG: 0, 15, 30, 45
                np.pi / 3, np.pi * 5 / 12, np.pi / 2, np.pi * 7 / 12,  # DEG: 60, 75, 90, 105
                np.pi * 2 / 3, np.pi * 3 / 4, np.pi * 5 / 6, np.pi * 11 / 12]]  # DEG: 120, 135, 150, 165


class ImageDatasetProcessing:
    # Processing process of an image dataset

    def __init__(self, dir_path, verbose=False) -> None:
        self.images = {}
        self.dataset_features = None
        self.verbose = verbose

        if self.verbose:
            print('Started loading images...')

        class_dirs = os.listdir(dir_path)
        for cdir in class_dirs:
            self.images[cdir] = []

            class_path = f'{dir_path}/{cdir}'
            file_list = os.listdir(class_path)

            for f in file_list:
                file_path = f'{class_path}/{f}'
                new_image = MriImage(name=f, category=cdir, file_path=file_path)

                self.images[cdir].append(new_image)

            if self.verbose:
                print(f'Finished loading images from {cdir} class.')

    # Generate LBP of all images using given parameters
    def generate_lbps(self, radius, n_points, method) -> pd.DataFrame:
        if self.verbose:
            print('Generating LBPs...')

        lbps_list = []

        for c in self.images:
            for img in self.images[c]:
                lbp_out = img.generate_lbps(radius, n_points, method)
                lbps_list.append(lbp_out)

            if self.verbose:
                print(f'LBPs of class {c} finished.')

        lbps_df = pd.concat(lbps_list)
        lbps_df = lbps_df.reset_index(drop=True)

        if self.verbose:
            print('Finished generating LBPs.')

        return lbps_df

    # Generate Zernike Moments of all images
    def generate_zernike_moments(self, radius) -> pd.DataFrame:
        if self.verbose:
            print('Generating Zernike Moments...')

        zm_list = []

        for c in self.images:
            for img in self.images[c]:
                zm_out = img.generate_zernike_moments(radius)
                zm_list.append(zm_out)

            if self.verbose:
                print(f'Zernike Moments of class {c} finished.')

        zm_df = pd.concat(zm_list)
        zm_df = zm_df.reset_index(drop=True)

        if self.verbose:
            print('Finished generating Zernike Moments.')

        return zm_df

    # Generate Gray Level Co-occurance Matrices for all images
    def generate_glcm(self, distances, angles, levels=256,
                      symmetric=True, normed=True) -> pd.DataFrame:
        if self.verbose:
            print('Generating GLCMs...')

        glcm_list = []

        for c in self.images:
            for img in self.images[c]:
                glcm_out = img.generate_glcm(distances, angles, levels, symmetric, normed)
                glcm_list.append(glcm_out)

            if self.verbose:
                print(f'GLCM of class {c} finished.')

        glcm_df = pd.concat(glcm_list)
        glcm_df = glcm_df.reset_index(drop=True)

        if self.verbose:
            print('Finished generating GLCMs.')

        return glcm_df

    def generate_features_for_dataset(self, lbp_params, zernike_params, glcm_params) -> pd.DataFrame:
        classes = [c for c in self.images for _ in self.images[c]]

        lbp = self.generate_lbps(lbp_params[0],
                                 lbp_params[1],
                                 lbp_params[2])
        print('---------------------------------------------------------------')
        glcm = self.generate_glcm(glcm_params[0],
                                  glcm_params[1], )
        print('---------------------------------------------------------------')
        zernike = self.generate_zernike_moments(zernike_params[0])

        self.dataset_features = lbp.join(glcm).join(zernike)

        # noinspection PyTypeChecker
        self.dataset_features.insert(0, 'Class', classes)

        return self.dataset_features

    def save_features_to_csv(self, path_to_dir, file_name) -> str:
        feature_df = self.dataset_features

        if feature_df is None:
            feature_df = self.generate_features_for_dataset(LBP_PARAMS,
                                                            ZM_PARAMS,
                                                            GLCM_PARAMS)

        if not os.path.exists(path_to_dir):
            subprocess.call(['mkdir', path_to_dir], shell=False)

        if file_name[-4:] != '.csv':
            file_name += '.csv'

        print(f'Features saved to file {file_name} in {path_to_dir}')

        path_to_file = f'{path_to_dir}/{file_name}'
        feature_df.to_csv(path_to_file)

        return path_to_file
