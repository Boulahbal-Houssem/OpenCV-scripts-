from sklearn.externals import joblib
from sklearn.svm import LinearSVC
import numpy as np
import hog
import cv2
import dataset
model = joblib.load("model/svm.cpickle")
hg = hog.HOG(orientations = 18 , pixelPerCell = (10,10),cellsPerBlock = (1,1), trasforme = True)
cap = cv2.VideoCapture(0)

while True:
	ret ,image = cap.read()
	image = cv2.imread("cellphone.png",1)
	gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
	blurred = cv2.GaussianBlur(gray , (5,5) , 0)
	edged = cv2.Canny(blurred , 30 ,150 )
	(_, cnt , _ ) = cv2.findContours(edged.copy() , cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
	cnts = sorted([(c , cv2.boundingRect(c)[0]) for c in cnt ] , key = lambda x: x[1])
	for (c, _ ) in cnts :
		(x,y,w,h) = cv2.boundingRect(c)
		if w >= 7 and h>=20:
			rio = gray[y:y+h,x:x+w]
			_,thresh =cv2.threshold(rio.copy(),0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
			thresh = dataset.deskew(thresh, 20)
			thresh = dataset.center_extent(thresh, (20, 20))
			hist = hg.describe(thresh)
			digit = model.predict([hist])[0]
			cv2.rectangle(image, (x, y), (x + w, y + h),
				(0, 255, 0), 1)
			cv2.putText(image, str(digit), (x - 10, y - 10),
				cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
			cv2.imshow("image", image)
	if cv2.waitKey(1) == ord('q'):
		break	
