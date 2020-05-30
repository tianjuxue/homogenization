from math import sqrt, pi, sin, cos
import numpy as np
import os
import random

def make_square(fid, n, x1, y1, x2, y2, dx):
	
	x = [x1, x2]
	y = [y1, y2]
	r = sqrt(2) * (x2 - x1)
	theta = [ (i/2.0 + 0.25) * pi for i in range(4)]

	index = n * 10000
    
	for i in range(4):
		print("Point(%d) = {%f, %f, %f, %f};" % (index + i + 1, 
			x1 + r * cos(theta[i]), y1 + r * sin(theta[i]), 0.0, dx), file=fid)

	for i in range(4):
		i4 = i + 2
		if i4 > 4:
			i4 = 1
		print("Line(%d) = {%d, %d};" % (index + i + 1, index + i + 1, index + i4), file=fid)

	print("Line Loop(%d) = {%d, %d, %d, %d};" % (index + 5, *[index + i + 1 for i in range(4)]), file=fid)
	return index + 5

def make_circle(fid, n, x1, y1, r, N, dx):
	
	theta = np.linspace(0, 1, N+1) * 2 * pi
	index = n * 10000
    
	for i in range(N):
		print("Point(%d) = {%f, %f, %f, %f};" % (index + i + 1, 
			x1 + r * cos(theta[i]), y1 + r * sin(theta[i]), 0.0, dx), file=fid)

	lineStr = ""

	for i in range(N):
		i4 = i + 2
		if i4 > N:
			i4 = 1
		print("Line(%d) = {%d, %d};" % (index + i + 1, index + i + 1, index + i4), file=fid)
		if i == 0:
			lineStr += str(index + i + 1)
		else:
			lineStr += ("," + str(index + i + 1))

	print("Line Loop(%d) = {%s};"%(index + N + 1, lineStr), file=fid)
	return index + N + 1

def make_clover_leaf(fid, n, x1, y1, a, N, NLeaf, dx):
	
	theta = np.linspace(0, 1, N+1) * 2 * pi
	index = n * 10000
    
	for i in range(N):
		s = NLeaf
		t = theta[i] 
		w = (0.5 - 2.0/s) * pi
		r = a * ( 0.75 + 0.25 * cos(s * (t-w)) )
		print("Point(%d) = {%f, %f, %f, %f};" % (index + i + 1, 
			x1 + r * cos(t), y1 + r * sin(t), 0.0, dx), file=fid)

	lineStr = ""

	for i in range(N):
		i4 = i + 2
		if i4 > N:
			i4 = 1
		print("Line(%d) = {%d, %d};" % (index + i + 1, index + i + 1, index + i4), file=fid)
		if i == 0:
			lineStr += str(index + i + 1)
		else:
			lineStr += ("," + str(index + i + 1))

	print("Line Loop(%d) = {%s};"%(index + N + 1, lineStr), file=fid)
	return index + N + 1


if __name__ == '__main__':

	fileDir = './'
	n = 20; L = 0.5; R = L*np.sqrt(0.5/np.pi); rd = 5e-3*L;
	fid = open(fileDir + 'geometry.geo', 'wt')
	length = n*L
	width = n*L
	line_loop_1 = make_square(fid, 1, length/2, width/2, length, width, 0.1)
	#line_loop_2 = make_circle(fid, 2, 0, 0, 0.8, 32, 0.1)
	line_loops = ""
	
	x = (np.arange(0, n) + 0.5) * L
	y = (np.arange(0, n) + 0.5) * L
	for i in range(n):
		for j in range(n):
			k = n * (i) + j + n
			x0 = x[i] + rd*(random.random()-0.5)
			y0 = y[j] + rd*(random.random()-0.5)
			line_loop = make_circle(fid, k, x0, y0, R, 32, 0.05)
			line_loops += (", " + str(line_loop))
	
	print("Plane Surface(%d) = {%s};"%(12, str(line_loop_1) + line_loops), file=fid)
	fid.close()
	# convert geofile to gmsh and then xml
	os.system('gmsh -v 1 -2 %sgeometry.geo' % (fileDir))
	os.system('dolfin-convert %sgeometry.msh %smesh-%d-%d.xml' % (fileDir, fileDir, n, n))
	
