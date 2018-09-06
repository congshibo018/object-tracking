import cv2
import numpy as np
import sys
def feature_descriptors(filename):
	cap = cv2.VideoCapture(filename)
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
			cv2.imshow("feature_descriptors",img2)
			k=cv2.waitKey(1)
		else:
			break
	k=cv2.waitKey(1)
	cv2.destroyAllWindows()

def matching(filename,num_obj):
	color = [(255,0,0),(0,255,0),(0,0,255)]
	cap = cv2.VideoCapture(filename)
	ret,frame = cap.read()
	detector = cv2.xfeatures2d.SIFT_create()
	#detector = cv2.xfeatures2d.SIFT_create(nfeatures=0, nOctaveLayers=3, contrastThreshold=0.04, edgeThreshold=10, sigma=1.6)
	#detector = cv2.xfeatures2d.SURF_create(hessianThreshold = 3000, nOctaves = 1, nOctaveLayers =  1, upright = True, extended = False)
	img1 = frame
	imCrops = []
	kp1_des1_list = []
	for i in range(num_obj):
		r = cv2.selectROI(img1)
		imCrop = img1[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
		imCrops.append(imCrop)
		kp1,des1 = detector.detectAndCompute(imCrop,None)
		kp1_des1_list.append([kp1,des1])
	MIN_MATCH_COUNT = 1
	#while 1:
	ret,frame = cap.read()
	img2 = frame
	if ret == True:
		for i in range(num_obj):

			kp2,des2 = detector.detectAndCompute(frame,None)
			FLANN_INDEX_KDTREE = 0
			index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
			search_params = dict(checks = 50)
			flann = cv2.FlannBasedMatcher(index_params, search_params)
			matches = flann.knnMatch(kp1_des1_list[i][1],des2,k=2)
			# store all the good matches as per Lowe's ratio test.
			good = []
			for m,n in matches:
			    if m.distance < 0.7*n.distance:
			        good.append(m)
			if len(good)>MIN_MATCH_COUNT:
			    src_pts = np.float32([ kp1_des1_list[i][0][m.queryIdx].pt for m in good ]).reshape(-1,1,2)
			    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

			    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,10.0)
			    matchesMask = mask.ravel().tolist()
			    h,w,c = imCrops[i].shape
			    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
			    if not M is None:
				    dst = cv2.perspectiveTransform(pts,M)

				    img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)
			else:
				matchesMask = None
			draw_params = dict(matchColor = color[i], # draw matches in green color
	                   singlePointColor = None,
	                   matchesMask = matchesMask, # draw only inliers
	                   flags = 2)
			img3 = cv2.drawMatches(imCrops[i],kp1_des1_list[i][0],img2,kp2,good,None,**draw_params)
			cv2.imshow('img3',img3)
			k = cv2.waitKey(0)
		k = cv2.waitKey(0)
	#else:
		#break
	cv2.destroyAllWindows()

def tracking(filename,num_obj):
	color = [(255,0,0),(0,255,0),(0,0,255)]
	cap = cv2.VideoCapture(filename)
	ret,frame = cap.read()
	img1 = frame
	rs = []
	for i in range(num_obj):
		r = cv2.selectROI(img1)
		rs.append(r)
		
	tracker_type = 'KCF'
	trackers = [cv2.TrackerKCF_create() for i in range(num_obj)]
	for i in range(len(rs)):
		status = trackers[i].init(frame,rs[i])
	while 1:
		ret,frame = cap.read()
		timer = cv2.getTickCount()
		if ret == True:
			for i in range(num_obj):
				tracker_status,r = trackers[i].update(frame)
				if tracker_status:
					p1 = (int(r[0]), int(r[1]))
					p2 = (int(r[0] + r[2]), int(r[1] + r[3]))
					cv2.rectangle(frame, p1, p2, color[i], 2, 1)
			fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
			cv2.putText(frame, tracker_type + " Tracker", (100,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50),2);
			cv2.putText(frame, "FPS : " + str(int(fps)), (100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2);
			cv2.imshow("Tracking", frame)
			k = cv2.waitKey(1)
		else:
			break
	k=cv2.waitKey(1)
	cv2.destroyAllWindows()

def trajectory(filename,num_obj):
	color = [(255,0,0),(0,255,0),(0,0,255)]
	cap = cv2.VideoCapture(filename)
	ret,frame = cap.read()
	detector = cv2.xfeatures2d.SIFT_create(nfeatures=0, nOctaveLayers=3, contrastThreshold=0.04, edgeThreshold=10, sigma=1.6)
	#detector = cv2.xfeatures2d.SURF_create(hessianThreshold = 3000, nOctaves = 1, nOctaveLayers =  1, upright = True, extended = False)
	img1 = frame
	rs = []
	for i in range(num_obj):
		r = cv2.selectROI(img1)
		rs.append(r)
	tracker_type = 'KCF'
	trackers = [cv2.TrackerKCF_create() for i in range(num_obj)]
	for i in range(len(rs)):
		status = trackers[i].init(frame,rs[i])
	traject_img = np.zeros(img1.shape)
	while 1:
		ret,frame = cap.read()
		timer = cv2.getTickCount()
		if ret == True:
			for i in range(num_obj):
				tracker_status,r = trackers[i].update(frame)
				if tracker_status:
					p1 = (int(r[0] + r[2]/2), int(r[1] + r[3]/2))
					p2 = int(r[2]/2)
					cv2.circle(traject_img, p1, 1, color[i], -1)
			fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
			cv2.imshow("trajectory", traject_img)
			k = cv2.waitKey(1)
		else:
			break
	k=cv2.waitKey(0)
	cv2.destroyAllWindows()

def main():
	if int(sys.argv[3]) not in [1,2,3]:
		print('Incorrect object number, try 1-3')
		return
	if sys.argv[1] == 'feature_descriptors':
		feature_descriptors(sys.argv[2])
	elif sys.argv[1] == 'matching':
		matching(sys.argv[2],int(sys.argv[3]))
	elif sys.argv[1] == 'tracking':
		tracking(sys.argv[2],int(sys.argv[3]))
	elif sys.argv[1] == 'trajectory':
		trajectory(sys.argv[2],int(sys.argv[3]))
	else:
		print('Try another command (feature_descriptor/ matching/ tracking/ trajectory)')
if __name__ == '__main__':
	main()
