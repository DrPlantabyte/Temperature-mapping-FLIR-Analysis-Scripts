import cv2, numpy, hashlib
import pandas
from numpy import ndarray
import pytesseract
from matplotlib import pyplot
from os import path
import os, sys, time
from datetime import datetime

from subprocess import call
# also requires ffmpeg CLI program installed and on the PATH variable

def main():
	show_plots = False
	upper_label_coord = (314, 23)
	lower_label_coord = (314, 234)
	coords_A1 =  [(22,42), (21,42), (44,56), (43,56)]
	coords_H12 = [(261,195), (260,195), (269,204), (268,202)]
	well_radius = 4
	i = -1
	data_dir = '../data/Thermocam Data'
	data_dirnames = os.listdir(data_dir)
	for dir in data_dirnames:
		i += 1
		dir_path = path.join(data_dir, dir)
		coord_A1 = coords_A1[i]
		coord_H12 = coords_H12[i]
		font_dir = '../resources/FLIR-Font'
		font_lut = load_fonts(font_dir)
		sample_name = path.basename(dir_path)
		global_max_temp = None
		global_min_temp = None
		files = [x for x in os.listdir(dir_path) if x.endswith('.jpg')]
		files.sort()
		time_series = []
		num = 0
		first_dt = None
		for f in files:
			img: ndarray = cv2.cvtColor(cv2.imread(path.join(dir_path, f)), cv2.COLOR_BGR2GRAY)
			max_temp = float(fetch_label(img, upper_label_coord, font_lut))
			min_temp = float(fetch_label(img, lower_label_coord, font_lut))
			if global_max_temp is None or global_max_temp < max_temp:
				global_max_temp = max_temp
			if global_min_temp is None or global_min_temp > min_temp:
				global_min_temp = min_temp
		print("global max temperature range: %s - %s" % (global_min_temp, global_max_temp))
		for f in files:
			num += 1
			year = int(f[4:8])
			month = int(f[8:10])
			day = int(f[10:12])
			hour = int(f[13:15])
			minute = int(f[16:18])
			second = int(f[19:21])
			dt = datetime(year, month, day, hour, minute, second)
			print(sample_name, 'Frame #',num,'Image @', dt.isoformat())
			if first_dt == None:
				first_dt = dt
			seconds = (dt - first_dt).total_seconds()
			img: ndarray = cv2.cvtColor(cv2.imread(path.join(dir_path, f)), cv2.COLOR_BGR2GRAY)
			max_temp = float(fetch_label(img, upper_label_coord, font_lut))
			min_temp = float(fetch_label(img, lower_label_coord, font_lut))
			therm = (img.astype(numpy.float64) / 255) * (max_temp - min_temp) + min_temp
			#
			temps = numpy.zeros((8,12)) # row, col
			for col in range(0, 12):
				for row in range(0, 8):
					well_xy = well_coord(row=row, col=col, coord_A1=coord_A1, coord_H12=coord_H12)
					T = therm[well_xy[1]-well_radius:well_xy[1]+well_radius, well_xy[0]-well_radius:well_xy[0]+well_radius].mean()
					temps[row][col] = T
			#
			time_series.append((dt, temps))
			# normalize image to fix scale across all images
			therm[0:4,0:4] = global_min_temp
			therm[0:4,4:8] = global_max_temp
			normalized_therm = (therm - global_min_temp) / (global_max_temp - global_min_temp)
			pyplot.imshow(normalized_therm)
			pyplot.set_cmap('inferno')
			pyplot.text(21, 21, dt.isoformat(sep=' '), color='black')
			pyplot.text(20, 20, dt.isoformat(sep=' '), color='white')
			dur = '%02d:%02d:%02d' % (int(seconds/3600), int(seconds/60) % 60, int(seconds) % 60)
			pyplot.text(201, 21, dur, color='black')
			pyplot.text(200, 20, dur, color='white')
			for col in range(0,12):
				for row in range(0,8):
					well_xy = well_coord(row=row, col=col, coord_A1=coord_A1, coord_H12=coord_H12)
					pyplot.plot(well_xy[0], well_xy[1], '+w')
					pyplot.text(well_xy[0]+3, well_xy[1]-1, '%.1f' % temps[row][col], color='black')
					pyplot.text(well_xy[0]+2, well_xy[1]-2, '%.1f' % temps[row][col], color='white')
			pyplot.savefig('%s_frame%03d.jpg' % (sample_name, num))
			if show_plots:
				pyplot.show()
			pyplot.clf()
		#
		time_series.sort()
		start_time = time_series[0][0]
		t = [(e[0]-start_time).total_seconds() for e in time_series]
		df_col_names = ['Time (s)']
		df_cols = [t]
		for col in range(0,12):
			data_series = numpy.asarray([e[1] for e in time_series])
			y = data_series[:,:,col].mean(axis=1)
			stddev = data_series[:,:,col].std(axis=1)
			df_col_names.append('Column %s' % (col+1))
			df_cols.append(y)
			df_col_names.append('Column %s std dev' % (col+1))
			df_cols.append(stddev)
			pyplot.plot(t, y, '-', label='Column %s' % (col+1))
			pyplot.fill_between(t, y-stddev, y+stddev, alpha=0.2)
		pandas.DataFrame(zip(*df_cols), columns=df_col_names).to_csv('%s Temp v Time.csv' % sample_name, index=False)
		pyplot.xlabel('Time (s)')
		pyplot.ylabel('Temperature (C) +/- 1 std. dev.')
		pyplot.grid()
		pyplot.legend()
		pyplot.savefig('%s Temp v Time.png' % sample_name)
		pyplot.savefig('%s Temp v Time.svg' % sample_name)
		if show_plots == True:
			pyplot.show()
		pyplot.clf()
		# make a video
		call(['ffmpeg' ,'-r' ,'10' ,'-start_number' ,'0' ,'-i' ,str(sample_name)+'_frame%3d.jpg' ,'%s Temp v Time - video.mp4' % sample_name])
	# done


def well_coord(col, row, coord_A1, coord_H12):
	dxdcol = (coord_H12[0] - coord_A1[0]) / 11
	dydrow = (coord_H12[1] - coord_A1[1]) / 7
	x = coord_A1[0] + (dxdcol * col)
	y = coord_A1[1] + (dydrow * row)
	return (int(x), int(y))
def load_fonts(font_dir):
	font_lut = {}
	for f in [f for f in os.listdir(font_dir) if str(f).endswith('.png')]:
		char = str(f).replace('.png', '')
		i = cv2.cvtColor(cv2.imread(path.join(font_dir, f)), cv2.COLOR_BGR2GRAY)
		font_lut[char] = i
		# font_lut[i] = char
	return font_lut

def fetch_label(img_grayscale, bottom_right, font_lut):
	bw_img = cv2.threshold( img_grayscale,128,255,cv2.THRESH_BINARY)[1]
	w = 1; h = 1
	while bw_img[bottom_right[1], bottom_right[0]-w] == 0 and w < 38:
		w += 1
	while bw_img[bottom_right[1]-h, bottom_right[0]] == 0 and h < 22:
		h += 1
	top_left = (bottom_right[0]-w+3, bottom_right[1]-h+3)
	preclip = bw_img[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
	# chop margins
	m_top = 0; m_bot = 0; m_lef = 0; m_rig = 0
	while preclip[m_top, :].max() == 0: m_top += 1
	while preclip[-m_bot-1, :].max() == 0: m_bot += 1
	while preclip[:, m_lef].max() == 0: m_lef += 1
	while preclip[:, -m_rig-1].max() == 0: m_rig += 1
	clip = preclip[m_top:-m_bot,m_lef:-m_rig]
	if clip.shape[0] > 12: # too tall
		clip = clip[-12:,:]
	if min(clip.shape) == 0:
		pyplot.imshow(preclip)
		pyplot.show()
		raise ValueError('Failed to clip label from image. Shape =', clip.shape)
	chops = []
	last_col = 0
	col = 1
	while col < clip.shape[1]:
		if clip[:,col].max() == 0 or col-last_col == 8:
			# chop character
			chops.append(clip[:,last_col:col].copy())
			while col < clip.shape[1] and clip[:,col].max() == 0:
				col += 1
			last_col = col
		else:
			col += 1
	if last_col < clip.shape[1]: chops.append(clip[:, last_col:col].copy())
	msg = ''
	for chop in chops:
		c = None
		for char in font_lut:
			char_img = font_lut[char]
			if char_img.tobytes() == chop.tobytes():
				c = char
				break
		if c == None:
			pyplot.imshow(chop)
			pyplot.show()
			pyplot.imshow(clip)
			pyplot.show()
			raise ValueError("Failed to identify character for image")
		else:
			msg += c
	# print(msg)
	return msg

main()
