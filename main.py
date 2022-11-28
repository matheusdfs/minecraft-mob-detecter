import cv2
import numpy
import time

from queryImage import QueryImage
from PIL import ImageGrab # for realtime screenshots
from matplotlib import pyplot as plt

MIN_MATCH_COUNT = 10
FLANN_INDEX_KDTREE = 1

def create_mask(queryImage, trainImage):
    hueValues = []

    queryImage = cv2.cvtColor(queryImage, cv2.COLOR_BGR2HLS)

    # Get the color palette of the sprite
    for row in queryImage:
        for element in row:
            h, l, s = element
            
            if h not in hueValues:
                hueValues.append(h)
            
    trainImageCopy = trainImage.copy()

    aux = trainImage[:,:,0]

    trainImageCopy = 1 if hueValues in aux else 0

    cv2.imshow('queryImageCopy', trainImageCopy)

    # print(colors)
    # print(len(colors))
    pass

def main():
    startTime = time.time()

    # Create object for queryImage
    queryImage = QueryImage(cv2.imread('default_mob_skins/minecraft_zombie_head.png', cv2.IMREAD_COLOR))

    # Open the ingame image
    trainImage = cv2.imread('zombie_ingame.jpg', cv2.IMREAD_COLOR)

    # Initiate SIFT detector
    sift = cv2.SIFT_create(contrastThreshold= 0.02, edgeThreshold= 17)

    # Generate the mask for the image
    mask = create_mask(queryImage.getOriginalImage(), trainImage)

    # Find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(queryImage.getImage(),None)
    kp2, des2 = sift.detectAndCompute(trainImage,None)

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