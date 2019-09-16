from skimage.feature import peak_local_max
from skimage.morphology import watershed
from scipy import ndimage
import numpy as np
import os

import cv2

def remove_samecircle(X,Y,R,x,y,r):
	if(r>70):
		return 1
	for i in range(len(X)):
		if(R[i]>70):
			continue
		d=np.sqrt(np.power(X[i]-x,2)+np.power(Y[i]-y,2))
		rr=np.max([R[i],r])
		if(d<rr):
			return 0
		if(r+R[i]-d>0):
			d1=r+R[i]-d
			if(d1/d)>0.7:
				return 0
	return 1

def get_points(image):
	# folder = 'Images/Color'
# for filename in os.listdir(folder):
# 	image_path=os.path.join(folder,filename)
# 	image=cv2.imread(folder+"/Color_20180109_092452_601.jpg")
	# lower_red = np.array([180,180,180])
	# upper_red = np.array([255,255,255])
	# mask = cv2.inRange(image, lower_red, upper_red)
	# cv2.imshow('origine image',image)
	height, width = image.shape[:2]
	# cv2.waitKey(0)

	shifted = cv2.pyrMeanShiftFiltering(image, 5, 21)
	gray = cv2.cvtColor(shifted, cv2.COLOR_BGR2GRAY)
	kernel = np.ones((5,5), np.uint8)
	# gray = cv2.erode(gray, kernel, iterations=3)
	thresh = cv2.threshold(gray, 0, 255,
		cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
	# cv2.imshow("Thresh", thresh)
	# cv2.waitKey(0)

	D = ndimage.distance_transform_edt(thresh)
	localMax = peak_local_max(D, indices=False, min_distance=10,
							  labels=thresh)

	# perform a connected component analysis on the local peaks,
	# using 8-connectivity, then appy the Watershed algorithm
	markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
	labels = watershed(-D, markers, mask=thresh)
	# print("[INFO] {} unique segments found".format(len(np.unique(labels)) - 1))
	# loop over the unique labels returned by the Watershed
	# algorithm
	num=1
	X=[]
	Y=[]
	R=[]
	points=[]
	for label in np.unique(labels):
		# if the label is zero, we are examining the 'background'
		# so simply ignore it
		if label == 0:
			continue

		# otherwise, allocate memory for the label region and draw
		# it on the mask
		mask = np.zeros(gray.shape, dtype="uint8")
		mask[labels == label] = 255

		# detect contours in the mask and grab the largest one
		cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
								cv2.CHAIN_APPROX_SIMPLE)[-2]
		c = max(cnts, key=cv2.contourArea)

		# draw a circle enclosing the object
		((x, y), r) = cv2.minEnclosingCircle(c)
		if(x<150 or x>850 or abs(y-height)<10):
			continue
		print(r)
		if( r<40):
			continue
		flag=remove_samecircle(X,Y,R,x,y,r)
		if(flag==0):
			continue
		X.append(x)
		Y.append(y)
		R.append(r)
		minx = max(int(x) - int(r*0.7), 0)
		miny = max(int(y) - int(r*0.7), 0)
		maxx = min(int(x) + int(r*0.7), width)
		maxy = min(int(y) + int(r*0.7), height)
		points.append((minx,miny,maxx,maxy))
		cv2.circle(image, (int(x), int(y)), int(r), (0, 255, 0), 2)
		cv2.putText(image, "#{}".format(num), (int(x) - 10, int(y)),
					cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
		# cv2.imshow("Output", image)
		num+=1
	return points
	# show the output image
	# cv2.imshow("Output", image)
	# cv2.waitKey(0)