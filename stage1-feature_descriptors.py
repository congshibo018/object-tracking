import cv2
import numpy as np

cap = cv2.VideoCapture('video_2_traffic_camera_15s.mp4')
ret,frame = cap.read()
detector = cv2.xfeatures2d.SIFT_create(nfeatures=0, nOctaveLayers=3, contrastThreshold=0.04, edgeThreshold=10, sigma=1.6)
#detector = cv2.xfeatures2d.SURF_create(hessianThreshold = 3000, nOctaves = 1, nOctaveLayers =  1, upright = True, extended = False)
img1 = frame
kp1,des1 = detector.detectAndCompute(frame,None)
while 1:
	ret,frame = cap.read()
	img2 = frame
	if ret == True:
		kp2,des2 = detector.detectAndCompute(frame,None)
		img2 = cv2.drawKeypoints(img2,kp2,img2)
		cv2.imshow("img2",img2)
		k=cv2.waitKey(1)
	else:
		break
k=cv2.waitKey(1)
cv2.destroyAllWindows()
