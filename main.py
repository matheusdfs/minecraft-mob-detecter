from typing import List, Tuple
import cv2
from cv2 import DMatch
import numpy as np
import time

from queryImage import QueryImage
from matplotlib import pyplot as plt

DESC_MATCH_MIN_RATIO = 0.6
MASK_RGB_MAX_DIST = 150


def create_mask(queryImage, trainImage):

    mediaB = 0
    mediaG = 0
    mediaR = 0

    for row in queryImage:
        for element in row:
            b, g, r = element

            mediaB += b
            mediaG += g
            mediaR += r

    media = np.array((mediaB, mediaG, mediaR)) / queryImage.size

    heigth = trainImage.shape[0]
    width = trainImage.shape[1]

    mask = trainImage.copy()

    for y in range(0, heigth):
        for x in range(0, width):
            dist = np.linalg.norm(trainImage[y, x] - media)

            if dist < MASK_RGB_MAX_DIST:
                mask[y, x] = [1, 1, 1]
            else:
                mask[y, x] = [0, 0, 0]

    return mask[:, :, 0]


def match_descriptors(queryDescs, trainDescs) -> List[Tuple[DMatch]]:
    matches = []
    for queryIdx, queryDesc in enumerate(queryDescs):
        best_dist_1 = 100000000
        best_dist_2 = 100000000
        train_idx_1 = 0

        for train_idx, trainDesc in enumerate(trainDescs):
            distance = np.linalg.norm(queryDesc - trainDesc)

            if distance < best_dist_1:
                best_dist_2 = best_dist_1

                best_dist_1 = distance
                train_idx_1 = train_idx

            elif distance < best_dist_2:
                best_dist_2 = distance

        if best_dist_1 < best_dist_2 * DESC_MATCH_MIN_RATIO:
            matches.append([DMatch(queryIdx, train_idx_1, best_dist_1)])
    return matches


def main():
    startTime = time.time()

    # Create object for queryImage
    queryImage = QueryImage(cv2.imread(
        'default_mob_skins/minecraft_zombie_head.png', cv2.IMREAD_COLOR))

    # Open the ingame image
    trainImage = cv2.imread('scenarios/zmb_skt_1.png', cv2.IMREAD_COLOR)

    # Initiate SIFT detector
    sift = cv2.SIFT_create(contrastThreshold=0.02, edgeThreshold=17)

    # Generate the mask for the image
    mask = create_mask(queryImage.getOriginalImage(), trainImage)

    # Find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(queryImage.getImage(), mask)
    kp2, des2 = sift.detectAndCompute(trainImage, mask)

    # Find descriptors matches
    matches = match_descriptors(des1, des2)
    print('Good matches', len(matches))

    match_img = cv2.drawMatchesKnn(queryImage.getImage(), kp1, trainImage, kp2,
                                   matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    print('Elapsed time:', time.time() - startTime, '(s)')

    cv2.imshow('match_img', match_img)
    cv2.waitKey()


if __name__ == '__main__':
    main()
