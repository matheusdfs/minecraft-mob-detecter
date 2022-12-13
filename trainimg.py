from cv2 import Mat


class TrainImgWrapper:
    __img = None
    __name = None

    def __init__(self, image, name):
        self.__name = name
        self.__image = image

    def get_img(self) -> Mat:
        return self.__image

    def get_name(self) -> str:
        return self.__name
