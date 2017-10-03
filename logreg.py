import numpy
import random

def accuracy(t, x, w, b):
	correct = 0
	for i in range(len(x)):
		ak = (x[i].dot(w) + b[i])/10000.0
		den = numpy.sum(numpy.exp(ak))/10000.0
		yk = numpy.exp(ak)/den
		if numpy.argmax(yk) == t[i]:
			correct += 1
	return float(correct/(len(x) + 1.0))

def logreg(training_data, validation_data, test_data, usps_test_date, iterations, learning_rate):
	training_x = training_data[0]
	training_t = training_data[1]
	validation_x = validation_data[0]
	validation_t = validation_data[1]
	test_x = test_data[0]
	test_t = test_data[1]
	n_samples = len(training_x)
	n_features = len(training_x[0])
	n_classes = 10
	yeta = learning_rate
	print iterations,		learning_rate


	
	# one-hot output vectors
	t_oh = []
	for tk in training_t:
		temp = [0]*n_classes
		temp[tk] = 1
		t_oh.append(temp)
	
	# initializing weights
	w = []
	for i in range(n_features):
		temp = []
		for j in range(n_classes):
			temp.append(random.random()*0.1)
		w.append(temp)

	# initializing bias terms
	b = []
	for i in range(n_samples):
		temp = []
		for j in range(n_classes):
			temp.append(random.random()*0.1)
		b.append(temp)

	# numpy arrays
	np_training_x = numpy.array(training_x)
	np_training_t = numpy.array(t_oh)
	np_validation_x = numpy.array(validation_x)
	np_validation_t = numpy.array(validation_t)
	np_test_x = numpy.array(test_x)
	np_test_t = numpy.array(test_t)
	np_w = numpy.array(w, dtype = 'float64')
	np_b = numpy.array(b)
	np_usps_test_date_x = numpy.array(usps_test_date[0])
	np_usps_test_date_t = numpy.array(usps_test_date[1])
	
	# print len(np_x), len(np_x[0])
	# print len(np_t_oh), len(np_t_oh[0])
	# print len(np_w), len(np_w[0])
	# print len(np_b), len(np_b[0])
	# print np_w.shape

	for j in range(iterations):
		print j		
		for i in range(n_samples):
			ak = np_training_x[i].dot(np_w) + np_b[i]
			den = numpy.sum(numpy.exp(ak))
			yk = numpy.exp(ak)/den
			np_w -= numpy.outer(np_training_x[i],(yk - np_training_t[i]))*yeta

	print "		training accuracy = ", accuracy(training_t, np_training_x, np_w, np_b)
	print "		validation accuracy = ", accuracy(validation_t, np_validation_x, np_w, np_b)
	print "		testing accuracy =", accuracy(test_t, np_test_x, np_w, np_b)
	print "		usps_test = ", accuracy(np_usps_test_date_t, np_usps_test_date_x, np_w, np_b)
