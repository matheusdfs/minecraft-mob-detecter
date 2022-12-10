import cv2
import numpy
import time

from queryImage import QueryImage
from matplotlib import pyplot as plt

MIN_MATCH_COUNT = 10
FLANN_INDEX_KDTREE = 1
MAX_DISTANCE = 150

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

    mediaB = mediaB/queryImage.size
    mediaG = mediaG/queryImage.size
    mediaR = mediaR/queryImage.size
            
    heigth = trainImage.shape[0]
    width = trainImage.shape[1]

    mask = trainImage.copy()

    for y in range(0, heigth):
        for x in range(0, width):
            b, g, r = trainImage[y, x]
            dist = numpy.sqrt((mediaB - b)**2 + (mediaG - g)**2 + (mediaR - r)**2)
            
            if dist < MAX_DISTANCE:
                mask[y, x] = [1, 1, 1]
            else:
                mask[y, x] = [0, 0, 0]
    
    return mask[:,:,0]


def main():
    startTime = time.time()

    # Create object for queryImage
    queryImage = QueryImage(cv2.imread('default_mob_skins/minecraft_zombie_head.png', cv2.IMREAD_COLOR))

    # Open the ingame image
    trainImage = cv2.imread('scenarios/zombie_ingame.jpg', cv2.IMREAD_COLOR)

    # Initiate SIFT detector
    sift = cv2.SIFT_create(contrastThreshold= 0.02, edgeThreshold= 17)

    # Generate the mask for the image
    mask = create_mask(queryImage.getOriginalImage(), trainImage)

    # Find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(queryImage.getImage(), mask)
    kp2, des2 = sift.detectAndCompute(trainImage, mask)

    img = queryImage.getImage().copy()

    img = cv2.drawKeypoints(queryImage.getImage() ,
                    kp1 ,
                    img ,
                    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imshow('img', img)

    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1,des2,k=2) #(tuple with matches)

    # store all the good matches as per Lowe's ratio test.
    good = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good.append(m)

    if len(good)>MIN_MATCH_COUNT:
        src_pts = numpy.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = numpy.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
        matchesMask = mask.ravel().tolist()

        h,w = queryImage.getImage().shape
        pts = numpy.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts,M) # 4 coordinates

        trainImage = cv2.polylines(trainImage,[numpy.int32(dst)],True,255,3, cv2.LINE_AA)

    else:
        print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
        matchesMask = None

    draw_params = dict(
                    matchColor       = (0,255,0), # draw matches in green color
                    singlePointColor = None,
                    matchesMask      = matchesMask, # draw only inliers
                    flags            = 2
                    )

    img3 = cv2.drawMatches(queryImage.getImage(),kp1,trainImage,kp2,good,None,**draw_params)

    print(time.time() - startTime)

    plt.imshow(img3, 'gray'),plt.show()

if __name__ == '__main__':
    main()