import os


def get_images_path_list(dir_path) -> []:
    # Get a list of paths of files from the given dir
    output_list = []
    category_dirs = os.listdir(dir_path)

    # Get categories
    for category in category_dirs:
        category_path = f'{dir_path}/{category}'
        file_list = os.listdir(category_path)

        # Get files from given category
        for file in file_list:
            file_path = f'{category_path}/{file}'
            output_list.append(file_path)

    return output_list
