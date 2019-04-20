from skimage import feature

class HOG:
	def __init__(self,orientations = 9 , pixelPerCell = (8,8),cellsPerBlock = (3,3), trasforme = True):
		self.orientations = orientations
		self.pixelPerCell = pixelPerCell
		self.cellsPerBlock = cellsPerBlock
		self.trasforme = trasforme
	def describe (self,image):
		hist = feature.hog(image, orientations=self.orientations,
				 pixels_per_cell=self.pixelPerCell,
				 cells_per_block=self.cellsPerBlock,
				 transform_sqrt = self.trasforme)
		return hist

