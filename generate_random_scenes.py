#!/usr/bin/env python

import sys
import os
import fnmatch
from tvtk.api import tvtk
from numpy.random import randint, rand, permutation
import numpy as np
import cv2
import json
import pylab
import scipy.misc as spm
import numpy.linalg as la

num_objects_in_scene = 8
model_center_square = 1.5

camera_height = 1.5
camera_height_delta = 0.5

camera_rotation_delta = 5

baseline = 0.075
focal_length = 585.0

width = 640
height = 480

max_num_trials = 100

def get_models_from_dir(models_dir):
	classes = os.listdir(models_dir)
	models = {}
	for c in classes:
		model_files = [ f for f in os.listdir(models_dir + '/' + c) if fnmatch.fnmatch(f,'*.vtk')]
		for m in model_files:
			r = tvtk.PolyDataReader()
			r.file_name = models_dir + '/' + c + '/' + m
			r.update()
			models[m] = tvtk.PolyDataMapper(input=r.output)
	return models

def get_models_names(models_dir):
	classes = os.listdir(models_dir)
	models = {}
	for c in classes:
		model_files = [ f for f in os.listdir(models_dir + '/' + c) if fnmatch.fnmatch(f,'*.vtk')]
		for m in model_files:
			#r = tvtk.PolyDataReader()
			#r.file_name = models_dir + '/' + c + '/' + m
			#r.update()
			models[m] = None #tvtk.PolyDataMapper(input=r.output)
	return models


def get_actor_from_models(models):
	actors = []
	scene_obj = {}
		
	m = models.keys()[randint(len(models))]
	mapper = models[m]

	scene_obj['model'] = m

	p = tvtk.Property(color=(1.0, 1.0, 1.0), lighting = False)
	actor = tvtk.Actor(mapper=mapper, property=p)

	x = (rand() - 0.5) * 2 * model_center_square
	y = (rand() - 0.5) * 2 * model_center_square
	theta = rand() * 360
	actor.position = [x, y, 0]
	actor.rotate_z(theta)

	#scene_obj['x'] = actor.position[0]
	#scene_obj['y'] = actor.position[1]
	#scene_obj['theta'] = actor.orientation[2]
	scene_obj['actor_matrix'] = actor.matrix.to_array().tolist()
	
	actors.append(actor)
	
	plane = tvtk.PlaneSource()
	plane_mapper = tvtk.PolyDataMapper(input=plane.output)

	p = tvtk.Property(color=(1.0, 0, 0), lighting = False)
	floor_actor = tvtk.Actor(mapper=plane_mapper, property=p)

	p = tvtk.Property(color=(0, 1.0, 0), lighting = False)
	wall_actor = tvtk.Actor(mapper=plane_mapper, property=p)

	wall_actor.rotate_x(90)
	wall_actor.scale = [10,6,1]
	wall_actor.position = [0, 3, 2]
	floor_actor.scale = [10,6,1]
	
	actors.append(wall_actor)
	actors.append(floor_actor)
	
	return actors, scene_obj

def get_render_window():
	ren = tvtk.Renderer(background=(0.0, 0.0, 0.0), automatic_light_creation=0)
	rw = tvtk.RenderWindow(size=(width,height))
	rw.off_screen_rendering = 0
	rw.add_renderer(ren)
	rw.multi_samples = 0
	ac = ren.active_camera
	ac.view_angle = 44.61
	#ren.reset_camera()
	return rw

def add_actors_to_render_window(rw, actors, scene_obj):
	actors_old = rw.renderers[0].actors
	for a in actors_old:
		rw.renderers[0].remove_actor(a)
	for a in actors:
		rw.renderers[0].add_actor(a)
	
	
	ac = rw.renderers[0].active_camera
	ac.position = [ 0, -3.0 , camera_height + (np.random.rand()-0.5)*2*camera_height_delta]
	ac.set_roll(np.random.randn()*camera_rotation_delta)
	m = ac.view_transform_matrix.to_array()
	#scene_obj['x'] -= pos[0]
	#scene_obj['y'] -= pos[1]
	#scene_obj['floor_plane'] = [ m[0,2], m[1,2], m[2,2], m[0,2]*m[0,3] + m[1,2]*m[1,3] + m[2,2]*m[2,3]]
	
	# Change from vtk coordinates to camera coordinates
	scene_obj['camera_matrix'] = np.dot(np.array([[1,0,0,0],[0,-1, 0,0],[0,0,-1,0],[0,0,0,1]]), m).tolist()
	
	#rw.renderers[0].add_actor(tvtk.AxesActor())
	#tm = rw.renderers[0].active_camera.view_transform_matrix.to_array()
	#tm = la.inv(np.dot(np.array([[1,0,0,0],[0,-1, 0,0],[0,0,-1,0],[0,0,0,1]]), tm))
	#t = tvtk.Transform()
	#t.set_matrix(np.ravel(tm))
	#a = tvtk.AxesActor()#tvtk.Actor(mapper=models[m])
	#a.user_transform = t
	#rw.renderers[0].add_actor(a)

	
	

def save_depth_image(rw, output_dir, i):
	narr = np.empty((height,width), dtype=np.float32)
	rw.get_zbuffer_data(0,0,width-1,height-1,narr)

	narr = np.flipud(narr)

	cr = rw.renderers[0].active_camera.clipping_range
	mask = (narr == 1.0)

	# Z buffer stores z values in log scale. Getting z coordinate in world coordinates.
	narr = (narr - 0.5) * 2.0
	narr = 2*cr[1]*cr[0]/(narr*(cr[1]-cr[0])-(cr[1]+cr[0]))
	narr = -narr

	# add gaussian noise multiplied by quadratic distance
	#narr = narr + narr * narr * np.random.normal(0, 0.01, (480,640))/8

	# Simulate finite disparity precision
	#disparity = baseline * focal_length / narr
	#disparity = np.round(disparity * 8)/8
	#narr = baseline * focal_length / disparity

	narr[mask] = 0
	narr = narr * 1000

	cv2.imwrite(output_dir + '/' + str(i) + '.depth.png', narr.astype(np.uint16))

def save_color_idx_image(rw, output_dir, i):
	img = np.empty((height * width * 4), dtype=np.uint8)
	rw.get_rgba_char_pixel_data(0,0,width-1,height-1,1,img)
	img = img.reshape((480,640,4))
	#img = img[:,:,0]
	img = np.flipud(img)
	#pylab.imshow(img)
	#pylab.show()
	#cv2.imwrite(output_dir + '/' + str(i) +'.color.png', img)
	spm.imsave(output_dir + '/' + str(i) +'.color.png', img)

def save_scene_list(scene_list, output_dir, i):
	f = open(output_dir + '/' + str(i) + '.json', 'w')
	f.write(json.dumps(scene_list, sort_keys=True, indent=4))
	f.close()
	

def gen_scenes(models_dir, output_dir, num_scenes):
	models = get_models_from_dir(models_dir)
	rw = get_render_window()
	
	for i in range(num_scenes):
		actors, scene_obj = get_actor_from_models(models)
		add_actors_to_render_window(rw, actors, scene_obj)
		rw.render()
		#rwi = tvtk.RenderWindowInteractor(render_window=rw)
		#rwi.initialize()
		#rwi.start()
		save_depth_image(rw, output_dir, i)
		save_color_idx_image(rw, output_dir, i)
		save_scene_list(scene_obj, output_dir, i)
	
	json.dump({'num_scenes': num_scenes}, open(output_dir + '/dataset.json', 'w'))

if __name__ == '__main__':
	gen_scenes(sys.argv[1], sys.argv[2], int(sys.argv[3]))
 
