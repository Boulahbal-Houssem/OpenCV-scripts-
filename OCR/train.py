from sklearn.externals import joblib
from sklearn.svm import LinearSVC
import hog
import dataset
import argparse


parser  = argparse.ArgumentParser("SVM training")
parser.add_argument("-d","--dataset",required = True,
			help ="Path to the database")
parser.add_argument("-m","--model", required= True,
			help = "Path to where the model will be stored")
args = vars(parser.parse_args())

(digits,target) = dataset.load_digits(args["dataset"])
data = []
hog = hog.HOG(orientations = 18 , pixelPerCell = (10,10),cellsPerBlock = (1,1), trasforme = True)
for image in digits:
	image = dataset.deskew(image,20)
	image = dataset.center_extent(image , (20,20) )
	hist = hog.describe(image)
	data.append(hist)

model = LinearSVC(random_state = 42) 
model.fit(data,target)
joblib.dump(model , args['model'])

