import cv2
import numpy

SCALE_MULTIPLIER = 30

class QueryImage:
    __original = None
    __imageResized = None

    def __init__(self, image):
        self.__original = image.astype(numpy.float32) / 255
        imageResized = cv2.resize(image, (image.shape[0] * 30, image.shape[1] * 30), interpolation= cv2.INTER_NEAREST)

        numpy.zeros([100,100,3], dtype = numpy.uint8)

    def getOriginalImage(self):
        return self.__original

    def getImage(self):
        return self.__imageResized

