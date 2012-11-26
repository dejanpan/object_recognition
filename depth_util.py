#!/usr/bin/env python
import numpy as np
from mayavi import mlab
from scipy.misc import imread
import os
import fnmatch

K_default = np.array([[575.8157348632812, 0.0, 314.5], [0.0, 575.8157348632812, 235.5], [ 0.0, 0.0, 1.0]], dtype=np.float32)
K_default_inv = np.linalg.inv(K_default)

def compute_grid(K_inv=K_default_inv, width=640, height = 480):
	grid = np.empty((height,width,3), dtype=np.float32)
	grid[:,:,1], grid[:,:,0] = np.mgrid[0:height,0:width]
	grid[:,:,2] = 1
	
	grid = np.dot(K_inv, grid.reshape(height*width, 3).T).T.reshape(height,width, 3)
	return grid
	
points_grid = compute_grid()

def compute_points(depth_image, K_inv=K_default_inv):
	points = points_grid * depth_image[:,:,np.newaxis] / 1000	
	return points

def compute_normals(points):
	h = points[1:,:-1,:] - points[:-1,:-1,:]
	v = points[:-1,1:,:] - points[:-1,:-1,:]
	n = np.cross(h,v)
	normalizer = np.sqrt(np.sum(n**2, axis=2))
	normalizer[normalizer == 0] = 1
	normals = np.zeros_like(points)
	normals[:-1,:-1,:] = n/normalizer[:,:,np.newaxis]
	return normals

def pshow(points, normals=None):
	if normals == None:
		mlab.points3d(points[:,:,0], points[:,:,1], points[:,:,2], color=(1,1,1), mode='point')
	else:
		mlab.quiver3d(points[:,:,0], points[:,:,1], points[:,:,2], normals[:,:,0], normals[:,:,1], normals[:,:,2], color=(1,1,1), mode='2darrow', scale_factor = 0.01)
	mlab.show()



def get_model_names(models_dir):
	classes = os.listdir(models_dir)
	models = []
	for c in classes:
		model_files = [ f for f in os.listdir(models_dir + '/' + c) if fnmatch.fnmatch(f,'*.vtk')]
		for m in model_files:
			models.append(m)
	return models
