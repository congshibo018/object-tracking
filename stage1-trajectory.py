import cv2
import numpy as np

def trajectory(filename):
	cap = cv2.VideoCapture(filename)
	ret,frame = cap.read()
	detector = cv2.xfeatures2d.SIFT_create(nfeatures=0, nOctaveLayers=3, contrastThreshold=0.04, edgeThreshold=10, sigma=1.6)
	#detector = cv2.xfeatures2d.SURF_create(hessianThreshold = 3000, nOctaves = 1, nOctaveLayers =  1, upright = True, extended = False)
	img1 = frame
	r = cv2.selectROI(img1)
	imCrop = img1[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
	tracker_type = 'KCF'
	tracker = cv2.TrackerKCF_create()
	tracker_status = tracker.init(frame,r)
	traject_img = np.zeros(img1.shape)
	while 1:
		ret,frame = cap.read()
		timer = cv2.getTickCount()
		if ret == True:
			tracker_status,r = tracker.update(frame)
			fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
			if tracker_status:
				p1 = (int(r[0] + r[2]/2), int(r[1] + r[3]/2))
				p2 = int(r[2]/2)
				cv2.circle(traject_img, p1, 1, (255,255,255), -1)

			cv2.imshow("trajectory", traject_img)
			k = cv2.waitKey(1)
		else:
			break
	k=cv2.waitKey(1)
	cv2.destroyAllWindows()
