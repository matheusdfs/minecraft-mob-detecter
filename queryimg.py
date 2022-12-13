import cv2
from cv2 import Mat
import numpy as np
import random


class QueryImgWrapper:
    __original = None
    __img_resized = None
    __mask = None
    __name = None

    def __init__(self, image, scale, name):
        self.__name = name
        self.__original = image.astype(np.float32) / 255
        image_resized = cv2.resize(
            image, (image.shape[0] * scale, image.shape[1] * scale), interpolation=cv2.INTER_NEAREST)

        self.__img_resized = np.zeros(
            (image_resized.shape[0] * 2, image_resized.shape[1] * 2, 3), dtype=np.uint8)
        self.__mask = np.zeros(
            (self.__img_resized.shape[0], self.__img_resized.shape[1]), dtype=np.uint8)

        h = self.__img_resized.shape[0]
        w = self.__img_resized.shape[1]

        for y in range(0, h):
            for x in range(0, w):
                if (1/4) * h < y < (3 / 4) * h and (1 / 4) * w < x < (3 / 4) * w:
                    self.__img_resized[y][x] = image_resized[int(
                        np.floor(y - (1/4) * h))][int(np.floor(x - (1/4) * w))]
                    self.__mask[y][x] = 255
                else:
                    self.__img_resized[y][x] = [random.randint(
                        0, 255), random.randint(0, 255), random.randint(0, 255)]
                    self.__mask[y][x] = 0

        kernel = np.ones((3, 3), np.uint8)
        self.__mask = cv2.dilate(self.__mask, kernel, iterations=1)

    def get_original(self) -> Mat:
        return self.__original

    def get_img(self) -> Mat:
        return self.__img_resized

    def get_mask(self) -> Mat:
        return self.__mask

    def get_name(self) -> str:
        return self.__name
