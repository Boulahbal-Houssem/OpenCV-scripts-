import cv2 
ret , I = cv2.imread('obama.png',1)
print(I)


img = cv2.imread(args["image",1])
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		# using FaceDetector class to detecte the face
		faceDetector = FaceDetector(args['face'])
		faces = faceDetector.detect(gray)
		# draw a box around faces
		for (x,y,w,h) in faces:
		    img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
		# draw image
		cv2.imshow('img',img)
		cv2.waitKey(0)
		cv2.destroyAllWindows()
