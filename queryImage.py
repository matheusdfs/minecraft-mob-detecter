import cv2

SCALE_MULTIPLIER = 30

class QueryImage:
    __original = None
    __imageResized = None

    def __init__(self, image):
        self.__original = image
        self.__imageResized = cv2.resize(image, (image.shape[0] * 30, image.shape[1] * 30), interpolation= cv2.INTER_NEAREST)

    def getOriginalImage(self):
        return self.__original

    def getImage(self):
        return self.__imageResized

