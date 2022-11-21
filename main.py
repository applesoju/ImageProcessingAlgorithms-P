from image_dataset import ImageDataset

RESOURCE_DIR_NAME = 'resources'

if __name__ == '__main__':
    mri_images_dataset = ImageDataset()
    mri_images_dataset.load_images(RESOURCE_DIR_NAME)
    mri_images_dataset.show_random_image()
