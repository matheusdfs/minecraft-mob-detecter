from cv2 import Mat
import cv2
import numpy as np

MASK_RGB_MAX_DIST = 40 / 255


def get_distances_to_color(img, clr):
    diff = img - clr
    sqr = diff**2
    s = sqr[:, :, 0] + sqr[:, :, 1] + sqr[:, :, 2]
    root = np.sqrt(s)
    return root


def create_mask(query_img, train_img) -> Mat:
    train_img = train_img.copy().astype(np.float32) / 255

    query_colors = set()
    for row in query_img:
        for elem in row:
            query_colors.add(tuple(elem))

    heigth = train_img.shape[0]
    width = train_img.shape[1]
    mask = np.zeros((heigth, width), dtype=np.uint8)

    for q_clr in query_colors:
        distances = get_distances_to_color(train_img, q_clr)
        mask = np.where(distances < MASK_RGB_MAX_DIST, 1.0, mask)

    mask = (mask * 255).astype(np.uint8)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)
    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask
