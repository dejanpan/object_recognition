#!/usr/bin/env python
import numpy as np
import scipy.misc as spm
import pylab
import json
import fern
from scipy.ndimage import find_objects, label, map_coordinates
from depth_util import compute_points, get_model_names


test_data_dir = 'test_data/'

classifier = fern.fern_classifier.from_file('fern.npz')

num_scenes = json.load(open(test_data_dir+'dataset.json'))['num_scenes']

correct = 0.0

for i in range(num_scenes):
	depth_image = spm.imread(test_data_dir+str(i)+'.depth.png', True)
	points = compute_points(depth_image)
	
	scene_obj = json.load(open(test_data_dir + str(i) + '.json'))
	camera_matrix = np.array(scene_obj['camera_matrix'], dtype=np.float32)
	actor_matrix = np.array(scene_obj['actor_matrix'], dtype=np.float32)
	m = scene_obj['model']
	
	plane_normal = camera_matrix[0:3,2]
	plane_point = camera_matrix[0:3,3]

	mask1 = np.abs(np.dot(points, plane_normal) -  np.dot(plane_point, plane_normal)) > 0.01
	mask2 = points[:,:,2] < 5.0
	mask = np.logical_and(mask1, mask2)

	sl = find_objects(mask)
	
	try:	
		obj_image = depth_image[sl[0]]
	except:
		pylab.imshow(depth_image)
		pylab.show()
		continue
	

	pred = classifier.classify(obj_image)
	if m == pred:
		correct += 1
	else:
		print 'Wrong classification of image', i
		print 'Prediction\t', pred
		print 'Truth\t', m
		print '\n'
		pylab.imshow(obj_image)
		pylab.show()


print 'Correctly classified ', correct/num_scenes

