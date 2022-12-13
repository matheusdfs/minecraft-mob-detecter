import time
from typing import List
import cv2
from loader import load_train_imgs, load_query_imgs
import numpy as np
from sklearn.cluster import DBSCAN
from mask import create_mask
from queryimg import QueryImgWrapper
from trainimg import TrainImgWrapper
from matcher import bf_match_descriptors

MATCH_MIN_RATIO = 0.6
TRAIN_IMG_DIR = 'scenarios/light/'
QUERY_IMG_DIR = 'mob_faces/'
RESULTS_DIR = 'results/'
QUERY_IMG_SCALE = 30
MIN_MATCH_CLUSTER = 3
MAX_MATCH_CLUSTER_DIST = 70
    

def get_bigger_cluster(points):
    clustering = DBSCAN(eps=MAX_MATCH_CLUSTER_DIST, min_samples=MIN_MATCH_CLUSTER).fit(points)
    labels = clustering.labels_
    unique, counts = np.unique(labels, return_counts=True)
    bigger_cluster_label_idx = counts.argmax()
    bigger_cluster_label = unique[bigger_cluster_label_idx]
    if bigger_cluster_label >= 0:
        flt = np.where(clustering.labels_ == bigger_cluster_label, True, False)
        return points[flt]
    else:
        return None


def draw_circle_if_mob_found(img, mob_name, matches, kp):
    train_idxs = [match[0].trainIdx for match in matches]
    keypoints = [kp[train_idx] for train_idx in train_idxs]

    if len(keypoints) > 0:
        points = cv2.KeyPoint_convert(keypoints)
        cluster = get_bigger_cluster(points)

        if cluster is not None:
            center = np.array((int(np.average(cluster[:,0])), int(np.average(cluster[:,1]))))
            radius = 70
            pink_color = (200, 0, 200)
            img = cv2.circle(img, center, radius, pink_color, 5)
            text_org = center + (-radius, -radius - 20)
            img = cv2.putText(img, mob_name, text_org, cv2.FONT_HERSHEY_COMPLEX, 1, pink_color, thickness=2)
        else:
            print(f'No {mob_name} found on image!')
    return img


def execute(sift, train_img: TrainImgWrapper, query_imgs: List[QueryImgWrapper]):
    start = time.time()
    result = train_img.get_img().copy()
    result_filename = train_img.get_name()
    for query_img in query_imgs:
        # Generate the mask for the image
        print(f'Creating mask for {train_img.get_name()} with {query_img.get_name()} query')
        mask = create_mask(query_img.get_original(), train_img.get_img())

        # Find the keypoints and descriptors with SIFT
        print('Detecting keypoints and descriptors')
        kp1, des1 = sift.detectAndCompute(query_img.get_img(), query_img.get_mask())
        kp2, des2 = sift.detectAndCompute(train_img.get_img(), mask)

        # Find descriptors matches
        print('Brute force matching descriptors')
        matches = bf_match_descriptors(des1, des2, MATCH_MIN_RATIO)
        print('Good matches', len(matches))

        print('Drawing matches')
        match_img = cv2.drawMatchesKnn(query_img.get_img(), kp1, train_img.get_img(), kp2,
                                    matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        mob_name = query_img.get_name()
        print(f'Drawing circle if {mob_name} found')
        result = draw_circle_if_mob_found(result, mob_name, matches, kp2)

        # cv2.namedWindow('match_img', cv2.WINDOW_NORMAL)
        # cv2.namedWindow('train_img', cv2.WINDOW_NORMAL)
        # cv2.resizeWindow('match_img', 1280, 720)
        # cv2.resizeWindow('train_img', 1280, 720)
        # cv2.imshow('match_img', match_img)
        # cv2.imshow('train_img', train_img.get_img())
        # cv2.imshow('mask', mask)
        # cv2.waitKey()
        cv2.imwrite(f'{RESULTS_DIR}/masks/{result_filename}-{mob_name}.png', mask)
        cv2.imwrite(f'{RESULTS_DIR}/matches/{result_filename}-{mob_name}.png', match_img)
    # cv2.namedWindow('result', cv2.WINDOW_NORMAL)
    # cv2.resizeWindow('result', 1280, 720)
    # cv2.imshow('result', result)
    # cv2.waitKey()
    cv2.imwrite(f'{RESULTS_DIR}{result_filename}.png', result)
    print(f'Elapsed time for {result_filename}: {time.time() - start} (s)')


def main():
    # Initiate SIFT detector
    sift = cv2.SIFT_create(contrastThreshold=0.01, edgeThreshold=20)

    # Load query images into QueryImgWrappers
    print(f'Loading query images from {QUERY_IMG_DIR}')
    query_imgs = load_query_imgs(QUERY_IMG_DIR, QUERY_IMG_SCALE)
    # query_map = {img.get_name() : img for img in query_imgs}

    # Open the ingame image
    print(f'Loading train images from {TRAIN_IMG_DIR}')
    train_imgs = load_train_imgs(TRAIN_IMG_DIR)

    for train_img in train_imgs:
        execute(sift, train_img, query_imgs)


if __name__ == '__main__':
    main()
