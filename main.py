from image_dataset import ImageDataset

RESOURCE_DIR_NAME = 'resources'

if __name__ == '__main__':
    mri_images_dataset = ImageDataset()
    mri_images_dataset.load_images(RESOURCE_DIR_NAME)
    # mri_images_dataset.show_random_image()
    # mri_images_dataset.show_image_from_category('ModerateDemented')
    # mri_images_dataset.show_random_histogram(scale='log')
    # mri_images_dataset.show_image_from_each_cat()
    # mri_images_dataset.show_hist_from_each_cat(scale='log')
    # mri_images_dataset.show_mean_histograms(scale='log')
