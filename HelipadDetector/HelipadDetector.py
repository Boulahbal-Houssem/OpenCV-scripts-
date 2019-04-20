import cv2
import numpy as np

def Harris(img):
	dst = cv2.cornerHarris(img,5,3,0.04)
	ret, dst = cv2.threshold(dst,0.1*dst.max(),255,0)
	dst = np.uint8(dst)
	ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)
	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
	corners = cv2.cornerSubPix(img,np.float32(centroids),(5,5),(-1,-1),criteria)
	return corners

def detecteH(corners,radius):
	__upperB = radius * 0.35
	__lowerB = radius * 0.5
	_upperB  = radius * 0.05
	_lowerB = radius * 0.20
	c = [i[0] for i in corners]
	r = [i[1] for i in corners]
	s = 0
	if (len(corners)==12):
		return True
	else:
		return False
	
def helipadDetector(frames):
	frame = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY);
	edges = cv2.Canny(frame,100,200)
	circles = cv2.HoughCircles(edges,cv2.HOUGH_GRADIENT,1,20,
		                    param1=50,param2=30,minRadius=30,maxRadius=250)
	circles = np.uint16(np.around(circles))
	s=0
	for i in circles[0,:]:
		mask = np.zeros((len(frames),len(frames[0]),1), np.uint8)
		cv2.circle(mask,(i[0],i[1]),int(i[2]*0.8),(255,255,255),-1)
		frameRio = cv2.multiply(edges,mask)
		cv2.imshow('s',frameRio)
		print("radius" +str(i[2]))
		corners = Harris(frameRio)
		print(detecteH(corners,i[1]))
		s+=1
		if s>5:
			break
		if cv2.waitKey(0) == ord('n'):
			pass
	if cv2.waitKey(0) == ord('q'):
		cv2.destroyAllWindows()


frames = cv2.imread('test.jpg',1)
helipadDetector(frames)


