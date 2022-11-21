import cv2


class MriImage:
    # Represents an MRI image

    def __init__(self, name, category, file_path) -> None:
        self.name = name
        self.category = category
        self.image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

    # Show an MRI image
    def show_image(self) -> None:
        # Generate a label for the image
        label = f'Name = {self.name}, ' \
                f'Category = {self.category}'

        # Display the image
        cv2.imshow(label, self.image)
        cv2.waitKey(0)
