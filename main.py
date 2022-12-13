import cv2
import numpy as np
from sklearn.cluster import DBSCAN
from mask import create_mask
from queryimg import load_query_imgs
from matcher import bf_match_descriptors
import time

MATCH_MIN_RATIO = 0.6
TRAIN_IMG_FP = 'scenarios/light/crp_1.png'
QUERY_IMG_DIR = 'mob_faces/'
QUERY_IMG_SCALE = 30
MIN_MATCH_CLUSTER = 3
MAX_MATCH_CLUSTER_DIST = 70


def analyze_results(query_img, train_img, train_mask, matches, kp1, kp2):
    

    print('Drawing matches')
    match_img = cv2.drawMatchesKnn(query_img.get_img(), kp1, train_img, kp2,
                                   matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    

def get_bigger_cluster(points):
    clustering = DBSCAN(eps=MAX_MATCH_CLUSTER_DIST, min_samples=MIN_MATCH_CLUSTER).fit(points)
    labels = clustering.labels_
    unique, counts = np.unique(labels, return_counts=True)
    bigger_cluster_label_idx = counts.argmax()
    bigger_cluster_label = unique[bigger_cluster_label_idx]
    flt = np.where(clustering.labels_ == bigger_cluster_label, True, False)
    points[flt]


def main():
    start_time = time.time()

    # Load query images into QueryImgWrappers
    print(f'Loading query images from {QUERY_IMG_DIR}')
    query_imgs = load_query_imgs(QUERY_IMG_DIR, QUERY_IMG_SCALE)
    query_map = {img.get_name() : img for img in query_imgs}

    # Open the ingame image
    print(f'Loading train image {TRAIN_IMG_FP}')
    train_img = cv2.imread(TRAIN_IMG_FP, cv2.IMREAD_COLOR)

    # Initiate SIFT detector
    sift = cv2.SIFT_create(contrastThreshold=0.02, edgeThreshold=17)

    # for query_img in query_imgs:
    query_img = query_map['creeper']
    # Generate the mask for the image
    print(f'Creating mask for {TRAIN_IMG_FP} with {query_img.get_name()} query')
    mask = create_mask(query_img.get_original(), train_img)

    # Find the keypoints and descriptors with SIFT
    print('Detecting keypoints and descriptors')
    kp1, des1 = sift.detectAndCompute(query_img.get_img(), query_img.get_mask())
    kp2, des2 = sift.detectAndCompute(train_img, mask)

    # Find descriptors matches
    print('Brute force matching descriptors')
    matches = bf_match_descriptors(des1, des2, MATCH_MIN_RATIO)
    print('Good matches', len(matches))


    print('Drawing matches')
    match_img = cv2.drawMatchesKnn(query_img.get_img(), kp1, train_img, kp2,
                                matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
    train_idxs = [match[0].trainIdx for match in matches]
    keypoints = [kp2[train_idx] for train_idx in train_idxs]
    points = cv2.KeyPoint_convert(keypoints)
    cluster = get_bigger_cluster(points)
    


    print('Elapsed time:', time.time() - start_time, '(s)')

    cv2.namedWindow('match_img', cv2.WINDOW_NORMAL)
    cv2.resizeWindow("match_img", 720, 1280)
    cv2.imshow('match_img', match_img)
    cv2.waitKey()


if __name__ == '__main__':
    main()
