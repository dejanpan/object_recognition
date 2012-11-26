#!/usr/bin/env python
import numpy as np
from scipy.ndimage import map_coordinates


class fern_classifier:
	@classmethod
	def from_parameters(cls, fern_depth, num_ferns, class_labels):

		fern = cls()
		fern.fern_depth = fern_depth
		fern.num_ferns = num_ferns
		fern.num_classes = len(class_labels)


		# Generate random tests
		fern.test = np.random.rand(2*num_ferns*fern_depth*2).reshape((2, 2, num_ferns, fern_depth)).astype(np.float32)
		# Create an array to store results
		fern.result = np.ones((num_ferns, 2**fern_depth, fern.num_classes), dtype=np.float32)
		# Generate array of labels
		fern.class_labels_array = np.array(class_labels)
		fern.normalized = False
		fern.labels_to_array_idx = { fern.class_labels_array[i]:i for i in range(fern.num_classes) }
		fern.normalizer = np.ones(fern.num_classes, dtype=np.float32) * (2**fern_depth)
		fern.fern_idx = np.arange(num_ferns, dtype = np.uint32)
		return fern

	@classmethod
	def from_file(cls, filename):
		fern = cls()
		npzfile = np.load(filename)
		fern.test = npzfile['test']
		fern.result = npzfile['result']
		fern.class_labels_array = npzfile['class_labels_array']

		fern.num_ferns = fern.test.shape[2]
		fern.fern_depth = fern.test.shape[3]
		fern.num_classes = fern.class_labels_array.shape[0]

		fern.fern_idx = np.arange(fern.num_ferns, dtype = np.uint32)
		fern.labels_to_array_idx = { fern.class_labels_array[i]:i for i in range(fern.num_classes) }

		fern.normalized = True
		return fern
		
	def add_training_example(self, obj_image, label):
		f = self.test.copy()
		f[0] *= obj_image.shape[0]
		f[1] *= obj_image.shape[1]
		r = map_coordinates(obj_image, f)
		r = r[0] > r[1]
	
		res = np.zeros(r.shape[0], np.int32)
		for i in range(r.shape[1]):
			res += r[:,i] << i
	
		model_idx = self.labels_to_array_idx[label]
		self.result[self.fern_idx, res, model_idx] += 1
		self.normalizer[model_idx] += 1

	def classify(self,obj_image):
		f = self.test.copy()
		f[0] *= obj_image.shape[0]
		f[1] *= obj_image.shape[1]
	
		r = map_coordinates(obj_image, f)
		r = r[0] > r[1]
	
		res = np.zeros(r.shape[0], np.int32)
		for i in range(r.shape[1]):
			res += r[:,i] << i
	
		d = self.result[self.fern_idx, res]

		dist = np.ones(self.num_classes, dtype=np.float64)/self.num_classes
		for i in range(d.shape[0]):
			dist *= d[i,:]
			dist /= dist.sum()

		return self.class_labels_array[dist.argmax()]
	
	def save_to_file(self, filename):
		if not self.normalized:
			self.result = self.result/self.normalizer
			self.normalized = True
		np.savez(filename, test=self.test, result=self.result, class_labels_array=self.class_labels_array)
		
