import sys

import cv2, numpy
from numpy import ndarray
import pytesseract
from matplotlib import pyplot
from os import path
import os, sys, time

def main():
	dir_path = 'D:\\CCHall\\OneDrive\\OneDrive - UTS\\Documents\\Projects\\2021\\Phenomics\\Thermocam Data\\2021-09-10'

	for f in [x for x in os.listdir(dir_path) if x.endswith('.jpg')]:
		# try:
		print(f)
		## note: cv2 images are [y][x][bgra] format
		img = cv2.cvtColor(cv2.imread(path.join(dir_path, f)), cv2.COLOR_BGR2GRAY)
		print(img.shape)
		cv2.imshow("whole image", img)
		upper_scale_img = clip_temperature_reading_window(img, (313,21))
		print(upper_scale_img.shape)
		cv2.imshow("upper", upper_scale_img)
		lower_scale_img = clip_temperature_reading_window(img, (313,232))
		cv2.imshow("lower", lower_scale_img)
		upper_str=pytesseract.image_to_string(upper_scale_img)
		lower_str=pytesseract.image_to_string(lower_scale_img)
		print("'%s', '%s'" % (upper_str, lower_str))
		cv2.waitKey(0)
		cv2.destroyAllWindows()
		# upper_value: float = float(upper_str.strip())
		# lower_value: float = float(lower_str.strip())
		# print('Range: %s - %s' % (lower_value, upper_value))
		# gs_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(numpy.float64)
		# imax = gs_img.max()
		# imin = gs_img.min()
		# val_img = ((gs_img - imin) / (imax - imin)) * (upper_value - lower_value) + lower_value
		# pyplot.imshow(val_img)
		# pyplot.set_cmap('gnuplot')
		# pyplot.show()
		# except ValueError as ex:
		# 	print(ex, file=sys.stderr)

def clip_temperature_reading_window(src_grayscale_img: ndarray, coord_xy: (int,int), fuzz=30, threshold=40, margin=10):
	## use a fill algorithm to draw a box around the text
	start_px = src_grayscale_img[coord_xy[1], coord_xy[0]] # cv2 images are stored as y,x not x,y
	searched = numpy.zeros_like(src_grayscale_img, dtype='b') # 0 for not searched, 1 for searched
	minx = coord_xy[0]
	maxx = coord_xy[0]
	miny = coord_xy[1]
	maxy = coord_xy[1]
	search_queue = set([coord_xy])

	# recursive search algorithm
	while(len(search_queue) > 0):
		s_coord = search_queue.pop()
		s_px = src_grayscale_img[s_coord[1], s_coord[0]]
		searched[s_coord[1], s_coord[0]] = 1
		if numpy.abs(float(start_px) - float(s_px)) < fuzz:
			## pixel is good
			minx = min(minx, s_coord[0])
			maxx = max(maxx, s_coord[0])
			miny = min(miny, s_coord[1])
			maxy = max(maxy, s_coord[1])
			## add unsearched neighbors to search queue
			left = (s_coord[0]-1, s_coord[1])
			right = (s_coord[0]+1, s_coord[1])
			up = (s_coord[0], s_coord[1]-1)
			down = (s_coord[0], s_coord[1]+1)
			for neighbor in [left, up, right, down]:
				already_searched = searched[neighbor[1], neighbor[0]]
				if already_searched == 0:
					search_queue.add(neighbor)
	## threshold the output
	clipped = cv2.threshold( src_grayscale_img[miny+1:maxy-1, minx+1:maxx-1],threshold,255,cv2.THRESH_BINARY)[1]
	## now pad it
	padded = numpy.ones((clipped.shape[0]+2*margin, clipped.shape[1]+2*margin), dtype=clipped.dtype) * 255
	padded[margin:margin+clipped.shape[0], margin:margin+clipped.shape[1]] = clipped
	## thicken the text
	a = padded.shape[0]; b = padded.shape[1]
	thickened = numpy.bitwise_and(padded[0:a-1,0:b-1] , padded[0:a-1,1:b])
	return thickened

main()
