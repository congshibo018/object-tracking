import cv2
import numpy as np

cap = cv2.VideoCapture('Video_sample_1.mp4')
ret,frame = cap.read()
detector = cv2.xfeatures2d.SIFT_create()
#detector = cv2.xfeatures2d.SIFT_create(nfeatures=0, nOctaveLayers=3, contrastThreshold=0.04, edgeThreshold=10, sigma=1.6)
#detector = cv2.xfeatures2d.SURF_create(hessianThreshold = 3000, nOctaves = 1, nOctaveLayers =  1, upright = True, extended = False)
img1 = frame
r = cv2.selectROI(img1)
imCrop = img1[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
kp1,des1 = detector.detectAndCompute(imCrop,None)
MIN_MATCH_COUNT = 2
while 1:
	ret,frame = cap.read()
	img2 = frame
	if ret == True:
		kp2,des2 = detector.detectAndCompute(frame,None)
		FLANN_INDEX_KDTREE = 0
		index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
		search_params = dict(checks = 50)
		flann = cv2.FlannBasedMatcher(index_params, search_params)
		matches = flann.knnMatch(des1,des2,k=2)
		# store all the good matches as per Lowe's ratio test.
		good = []
		for m,n in matches:
		    if m.distance < 0.7*n.distance:
		        good.append(m)
		if len(good)>MIN_MATCH_COUNT:
		    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
		    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

		    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
		    matchesMask = mask.ravel().tolist()
		    h,w,c = imCrop.shape
		    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
		    if not M is None:
			    dst = cv2.perspectiveTransform(pts,M)

			    img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)
		else:
			matchesMask = None
		draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)
		img3 = cv2.drawMatches(imCrop,kp1,img2,kp2,good,None,**draw_params)
		cv2.imshow('img3',img3)
		k = cv2.waitKey(1)
	else:
		break
cv2.destroyAllWindows()
