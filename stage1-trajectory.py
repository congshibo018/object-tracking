import cv2
import numpy as np

cap = cv2.VideoCapture('Video_sample_1.mp4')
ret,frame = cap.read()
detector = cv2.xfeatures2d.SIFT_create(nfeatures=0, nOctaveLayers=3, contrastThreshold=0.04, edgeThreshold=10, sigma=1.6)
#detector = cv2.xfeatures2d.SURF_create(hessianThreshold = 3000, nOctaves = 1, nOctaveLayers =  1, upright = True, extended = False)
img1 = frame
r = cv2.selectROI(img1)
imCrop = img1[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
tracker_type = 'KCF'
tracker = cv2.TrackerKCF_create()
tracker_status = tracker.init(frame,r)
while 1:
	ret,frame = cap.read()
	timer = cv2.getTickCount()
	if ret == True:
		tracker_status,r = tracker.update(frame)
		fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
		if tracker_status:
			p1 = (int(r[0]), int(r[1]))
			p2 = (int(r[0] + r[2]), int(r[1] + r[3]))
			cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)
		else :
			# Tracking failure
			cv2.putText(frame, "Tracking failure detected", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)
		cv2.putText(frame, tracker_type + " Tracker", (100,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50),2);
		cv2.putText(frame, "FPS : " + str(int(fps)), (100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2);
		cv2.imshow("Tracking", frame)
		k = cv2.waitKey(1)
	else:
		break
k=cv2.waitKey(1)
cv2.destroyAllWindows()
