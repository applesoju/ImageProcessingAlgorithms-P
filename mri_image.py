import cv2


class MriImage:
    def __init__(self, name, category, file_path) -> None:
        self.name = name
        self.category = category
        self.image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

    def show_image(self) -> None:
        label = f'Random image\n' \
                f'Category = {self.category}\n' \
                f'Name={self.name}'

        cv2.imshow(label, self.image)
        cv2.waitKey(0)
