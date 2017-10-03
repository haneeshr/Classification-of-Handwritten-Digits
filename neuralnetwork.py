
import numpy
import random

def h(zj):
	return 1/(1+numpy.exp(-zj))

def hp(zj):
	return zj.dot((1 - zj))

def accuracy(t, x ,w1, w2, b1, b2):
	correct = 0
	for i in range(len(x)):
		aj = x[i].dot(w1) + b1[i]
		zj = h(aj)
		ak = zj.dot(w2) + b2[i]
		den = numpy.sum(numpy.exp(ak))
		yk = numpy.exp(ak)/den
		if numpy.argmax(yk) == t[i]:
			correct += 1
	# print zj
	return float(correct/(len(x) + 1.0))

def neuralnetwork(training_data, validation_data, test_data, usps_test_date):
	training_x = training_data[0]
	training_t = training_data[1]
	validation_x = validation_data[0]
	validation_t = validation_data[1]
	test_x = test_data[0]
	test_t = test_data[1]
	n_samples = len(training_x)
	n_features = len(training_x[0])
	n_classes = 10
	n_hidden_units = 100
	yeta = 0.001
	iterations = 50

	print n_hidden_units, yeta, iterations
	# print yeta
	# print iterations
	# one-hot outputs
	t_oh = []
	for tk in training_t:
		temp = [0]*n_classes
		temp[tk] = 1
		t_oh.append(temp)

	# initializing hidden layer weights(features x basis)
	w1 = []
	for i in range(n_features):
		temp = []
		for j in range(n_hidden_units):
			temp.append(random.random()*0.01)
		w1.append(temp)

	# initializing hidden layer bias terms(samples x basis)
	b1 = []
	for i in range(n_samples):
		temp = []
		for j in range(n_hidden_units):																																																																												
			temp.append(random.random()*0.01)
		b1.append(temp)

	# initializing weights(basis x classes)
	w2 = []
	for i in range(n_hidden_units):
		temp = []
		for j in range(n_classes):
			temp.append(random.random()*0.01)
		w2.append(temp)


	# initializing bias terms(samples x classes)
	b2 = []
	for i in range(n_samples):
		temp = []
		for j in range(n_classes):																																																																												
			temp.append(random.random()*0.01)
		b2.append(temp)



	np_w1 = numpy.array(w1)
	np_b1 = numpy.array(b1)							
	np_w2 = numpy.array(w2, dtype = 'float64')
	np_b2 = numpy.array(b2)							
	np_training_x = numpy.array(training_x)
	np_training_t = numpy.array(t_oh)
	np_validation_x = numpy.array(validation_x)
	np_validation_t = numpy.array(validation_t)
	np_test_x = numpy.array(test_x)
	np_test_t = numpy.array(test_t)
	np_usps_test_date_x = numpy.array(usps_test_date[0])
	np_usps_test_date_t = numpy.array(usps_test_date[1])
	
	# print np_w1.shape
	# print np_b1.shape
	# print np_w2.shape
	# print np_b2.shape
	# print np_z.shape

	for j in range(iterations):
		print j
		for i in range(n_samples):
			aj = np_training_x[i].dot(np_w1) + np_b1[i]
			zj = h(aj)
			ak = zj.dot(np_w2) + np_b2[i]
			yk = numpy.exp(ak)/numpy.sum(numpy.exp(ak))
			delk = yk - np_training_t[i]
			delj = hp(zj)*(np_w2.dot(delk))
			delE2 = numpy.outer(delk,zj)
			delE1 = numpy.outer(np_training_x[i], delj)
			np_w2 -= yeta*delE2.transpose()
			np_w1 -= yeta*delE1


	print "		training = ", accuracy(training_t, np_training_x, np_w1, np_w2, np_b1, np_b2)
	print "		validation = ", accuracy(validation_t, np_validation_x, np_w1, np_w2, np_b1, np_b2)
	print "		test = ", accuracy(test_t, np_test_x, np_w1, np_w2, np_b1, np_b2)
	print "		usps_test = ", accuracy(np_usps_test_date_t, np_usps_test_date_x, np_w1, np_w2, np_b1, np_b2)