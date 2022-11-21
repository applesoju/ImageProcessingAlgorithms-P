import matplotlib.pyplot as plt
import numpy as np
import cv2


class MriImage:
    def __init__(self, name, category, file_path):
        self.name = name
        self.category = category
        self.image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

    def show_image(self):
        cv2.imshow('image', self.image)
        cv2.waitKey(0)
