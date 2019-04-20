import cv2
import covermatcher
import coverdescriptor
import glob
import csv

db_path ="books.csv"
db ={}
query = ""
for l in csv.reader(open(db_path)):
	db[l[0]] = l[1:]

useHamming = True
ratio = 0.7
minMatches = 40
cd = CoverDescriptor(useSIFT = useSIFT)
cv = CoverMatcher(cd, glob.glob(db_path + "/*.png"),ratio = ratio, minMatches = minMatches, useHamming =useHamming)
queryImage = cv2.imread(query)
gray = cv2.cvtColor(queryImage, cv2.COLOR_BGR2GRAY)
(queryKps, queryDescs) = cd.describe(gray)
	results = cv.search(queryKps, queryDescs)
cv2.imshow("Query", queryImage)	
