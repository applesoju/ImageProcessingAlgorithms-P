import cv2
import mahotas
import numpy as np
import pandas as pd
from skimage import feature


class MriImage:
    # Represents an MRI image

    def __init__(self, name, category, file_path) -> None:
        self.name = name
        self.category = category
        self.image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

    # Creates a Local Binary Pattern descriptor of the image
    def generate_lbps(self, radius, n_points, method) -> pd.DataFrame:
        if type(radius) is not list:
            radius = [radius]

        lbps_dict = {}

        for r in radius:
            lbp = feature.local_binary_pattern(self.image, n_points, r, method=method)
            n_vals = int(lbp.max() + 1)

            for val in range(n_vals):
                val_count = np.count_nonzero(lbp == val) / lbp.size
                column_name = f'lbp{n_points:02d}-{r:02d}{val:02d}'

                lbps_dict[column_name] = val_count

        lbp_df = pd.DataFrame(data=lbps_dict, index=[0])

        return lbp_df

    # Creates a Zernike Moments of the image
    def generate_zernike_moments(self, radius) -> pd.DataFrame:
        if type(radius) is not list:
            radius = [radius]

        zernike_dict = {}

        for r in radius:
            zernike = mahotas.features.zernike_moments(self.image, r)
            column_names = [f'zernike{r:02d}-{i:02d}' for i in range(len(zernike))]

            for cn, zm in zip(column_names, zernike):
                zernike_dict[cn] = zm

        zm_df = pd.DataFrame(data=zernike_dict, index=[0])

        return zm_df

    def generate_glcm(self, dists, angles, levels, sym, norm) -> pd.DataFrame:
        if type(dists) is not list:
            dists = [dists]

        if type(angles) is not list:
            angles = [angles]

        props = ['dissimilarity', 'correlation', 'homogeneity', 'contrast', 'ASM', 'energy']

        glcm = feature.graycomatrix(
            image=self.image,
            distances=dists,
            angles=angles,
            levels=levels,
            symmetric=sym,
            normed=norm
        )

        glcms_dict = {}

        for prop in props:
            prop_vals = feature.graycoprops(glcm, prop)

            for d, dist in enumerate(dists):
                for a, angle in enumerate(angles):
                    angle_deg = int(np.rad2deg(angle))

                    prop_value = prop_vals[d, a]
                    column_name = f'{prop}-{dist:02d}{angle_deg:03d}'

                    glcms_dict[column_name] = prop_value

        glcm_df = pd.DataFrame(data=glcms_dict, index=[0])

        return glcm_df
