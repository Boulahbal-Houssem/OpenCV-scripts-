import cv2
import hog
def main():
	charDetector = hog.HOG()
	cap = cv2.VideoCapture(0)
	while True:
		ret , frame = cap.read()
		gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
		cv2.imshow('s',gray)
		charDetector.describe(gray)
		
		if cv2.waitKey(1) == ord('q'):
			break 


if __name__ == "__main__":
	main()
