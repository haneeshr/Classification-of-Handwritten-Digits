import numpy
import glob
import cPickle
import gzip
import logreg
import neuralnetwork
import tensor
from PIL import Image

new_width  = 28
new_height = 28

# filename = 'mnist.pklself.gz'
# f = gzip.open(filename, 'rb')

# training_data, validation_data, test_data = cPickle.load(f)
# hiddenunits = [200, 300, 400, 500, 600]
# learning_rates = [0.01, 0.03, 0.05, 0.07, 0.09, 0.1]
# iterations = [1000, 2000, 5000, 10000, 20000 ]
# f.close()

usps_test_data = [[],[]]
for j in range(10):
	for image in glob.glob("Numerals/" + str(j) + "/*.png"):
		img = Image.open(image)
		img = img.resize((new_height, new_width))
		pix = numpy.array(img.getdata(), dtype = 'float64')
		pix = [1]*784 - (pix/([255]*784))
		usps_test_data[0].append(pix)
		usps_test_data[1].append(j)
		

# print numpy.array(usps_test_data[0][0])
# print training_data[0][0]
# print numpy.array(usps_test_data[1])

print "LOGISTIC REGRESSION"
logreg.logreg(training_data, validation_data, test_data, usps_test_data, 10, 0.0005)

print "SINGLE LAYER NEURAL NETWORKS"
neuralnetwork.neuralnetwork(training_data, validation_data, test_data, usps_test_data)

print "CONVOLUTIONAL NEURAL NETWORKS"
tensor.tensorflow(usps_test_data[0], usps_test_data[1])
