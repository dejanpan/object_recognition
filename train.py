#!/usr/bin/env python
import sys
import json
import fern
import numpy as np
import scipy.misc as spm
from scipy.ndimage import find_objects, label, map_coordinates
from depth_util import compute_points, get_model_names
import pylab


models_dir = 'models/'
train_data_dir = 'training_data/'

models = get_model_names(models_dir)

classifier = fern.fern_classifier.from_parameters(fern_depth=12, num_ferns=50, class_labels=models)

num_scenes = json.load(open(train_data_dir+'dataset.json'))['num_scenes']


for i in range(num_scenes):
	depth_image = spm.imread(train_data_dir+str(i)+'.depth.png', True)
	points = compute_points(depth_image)
	
	scene_obj = json.load(open(train_data_dir + str(i) + '.json'))
	camera_matrix = np.array(scene_obj['camera_matrix'])
	actor_matrix = np.array(scene_obj['actor_matrix'])
	m = scene_obj['model']
	
	plane_normal = camera_matrix[0:3,2]
	plane_point = camera_matrix[0:3,3]

	d = np.dot(plane_point, plane_normal)
	dist = np.dot(points, plane_normal)
	mask1 = np.abs(dist -  d) > 0.05
	mask2 = depth_image < 5000.0
	mask = np.logical_and(mask1, mask2)

	sl = find_objects(mask)
	try:	
		obj_image = depth_image[sl[0]]
		
	except:
		continue
	
	classifier.add_training_example(obj_image, m)


classifier.save_to_file('fern.npz')


