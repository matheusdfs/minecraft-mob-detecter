import os
import sys
from typing import List

import cv2
from cv2 import Mat
from trainimg import TrainImgWrapper
from queryimg import QueryImgWrapper


def load_query_imgs(directory, scale) -> List[QueryImgWrapper]:
    imgs = []
    for filename in os.listdir(directory):
        filepath = directory + filename
        img = cv2.imread(filepath, cv2.IMREAD_COLOR)

        if img is None:
            print(f'Error opening image: {filepath}\n')
            sys.exit()

        name = filename.split('.')[0]
        imgs.append(QueryImgWrapper(img, scale, name))
    return imgs


def load_train_imgs(directory) -> List[Mat]:
    imgs = []
    for filename in os.listdir(directory):
        filepath = directory + filename
        img = cv2.imread(filepath, cv2.IMREAD_COLOR)

        if img is None:
            print(f'Error opening image: {filepath}\n')
            sys.exit()

        name = filename.split('.')[0]
        imgs.append(TrainImgWrapper(img, name))
    return imgs
