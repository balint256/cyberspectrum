#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  spectrum_viewer.py
#  
#  Part of: https://github.com/balint256/cyberspectrum
#  
#  Copyright 2014 Balint Seeber <balint256@gmail.com>
#  
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 2 of the License, or
#  (at your option) any later version.
#  
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#  
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software
#  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#  MA 02110-1301, USA.
#  
#  

import sys, os, datetime
from optparse import OptionParser

import numpy

from fft_tools import *
from realtime_graph import *
from utils import *

class CaptureFile():
	def __init__(self, path, rate, config_name, capture_time, idx, chan, freq, gain, antenna):
		self.path = path
		self.rate = rate
		self.config_name = config_name
		self.capture_time = capture_time
		self.idx = idx
		self.chan = chan
		self.freq = freq
		self.gain = gain
		self.antenna = antenna
	def __str__(self):
		# Not showing: path, config_name, chan
		return "%s %05d: %d sps, %d Hz, %d, %s" % (self.capture_time, self.idx, self.rate, self.freq, self.gain, self.antenna)

def _stitch_array(orig, new, overlap):
	if overlap == 0:
		return numpy.concatenate((orig, new))
	
	new_part = new[:overlap]
	orig_part = orig[-overlap:]
	part = numpy.add(orig_part, new_part)
	return numpy.concatenate((orig[:-overlap], part, new[overlap:]))

def main():
	parser = OptionParser(usage="%prog: [options] files...")
	
	parser.add_option("-s", "--samp-rate", type="float", default=1e6, help="Sample rate (Hz) [default=%default]")
	parser.add_option("-c", "--channel", type="string", default=None, help="Plot single channel [default=%default]")
	parser.add_option("-l", "--fft-length", type="int", default=16384, help="FFT length [default=%default]")
	parser.add_option("-L", "--lower-limit", type="float", default=-130, help="Lower amplitude limit [default=%default]")
	parser.add_option("-U", "--upper-limit", type="float", default=0, help="Upper amplitude limit [default=%default]")
	parser.add_option("-t", "--type", type="string", default="c8", help="Numpy file type [default=%default]")
	parser.add_option("-S", "--start-idx", type="int", default=0, help="Capture list index start [default=%default]")
	parser.add_option("-E", "--end-idx", type="int", default=-1, help="Capture list index end [default=%default]")
	parser.add_option("-o", "--overlap", type="float", default=None, help="Overlap amount [default=%default]")
	parser.add_option("-f", "--start-freq", type="float", default=None, help="Lower frequency limit [default=%default]")
	parser.add_option("-e", "--stop-freq", type="float", default=None, help="Upper frequency limit [default=%default]")
	
	(options, args) = parser.parse_args()
	
	print "Files:", len(args)
	
	channels = {}
	for arg in args:
		name = os.path.basename(arg)
		name = os.path.splitext(name)[0]
		parts = name.split("-")
		time_parts = parts[1].split("_")
		sec = float(time_parts[5])
		whole_sec = int(sec)
		us = int((sec - whole_sec) * 1e6)
		capture_time = datetime.datetime(int(time_parts[0]), int(time_parts[1]), int(time_parts[2]), int(time_parts[3]), int(time_parts[4]), whole_sec, us)
		channel_count = int(parts[3])
		freq = float(parts[4])
		antenna = parts[6].replace("_", "/")
		idx = int(parts[7])
		cap = CaptureFile(arg, options.samp_rate, parts[0], capture_time, int(parts[2]), idx, freq, float(parts[5]), antenna)
		if idx in channels.keys():
			channels[idx] += [cap]
		else:
			channels[idx] = [cap]
	
	if False:
		for channel in channels.keys():
			print channel
			channel_caps = channels[channel]
			for cap in channel_caps:
				print cap
	
	if options.channel is None or len(options.channel) == 0:
		channels_to_show = channels.keys()
	else:
		channels_to_show = map(int, options.channel.split(','))
	
	channel_count = len(channels_to_show)
	
	###############################
	
	font = {
		#'family' : 'normal',
		#'weight' : 'bold',
		'size'   : 10
	}
	
	matplotlib.rc('font', **font)

	padding = 0.05
	spacing = 0.1
	figure_width = 8
	figure_height = 10
	
	if channel_count > 2:
		channel_pos = 220
		figure_width = figure_width * 2
	elif channel_count == 2:
		channel_pos = 210
	else:
		channel_pos = 110
	
	figsize = (figure_width, figure_height)
	padding = {'wspace':spacing,'hspace':spacing,'top':1.-padding,'left':padding,'bottom':padding,'right':1.-padding}
	fft_graph = realtime_graph(title="FFT", show=True, manual=True, redraw=False, figsize=figsize, padding=padding)
	fft_channel_graphs = {}
	
	pos_count = 0
	y_limits = (options.lower_limit, options.upper_limit)
	for channel_idx in channels_to_show:
		#if channel_count > 2:
		#    pos_offset = ((pos_count % 2) * 2) + (pos_count / 2) + 1   # Re-order column-major
		#else:
		pos_offset = pos_count + 1
		subplot_pos = (channel_pos + pos_offset)
		
		channel = channels[channel_idx]
		fft_avg, fft_min, fft_max = numpy.array([]), numpy.array([]), numpy.array([])
		cnt = 0
		if options.end_idx == -1:
			options.end_idx = len(channel) - 1
		#for cap in channel:
		freq_min = None
		freq_max = None
		freq_last = None
		freq_line = []
		freq_center_line = []
		for cap_idx in range(options.start_idx, options.end_idx+1):
			cnt += 1
			cap = channel[cap_idx]
			freq_bottom = cap.freq - options.samp_rate/2
			freq_top = cap.freq + options.samp_rate/2
			
			if options.start_freq is not None and freq_top < options.start_freq:
				continue
			elif options.stop_freq is not None and freq_bottom > options.stop_freq:
				break
			
			if freq_min is None: freq_min = cap.freq
			else:
				if cap.freq <= freq_min:
					raise Exception("Out-of-order captures")
			if freq_max is None: freq_max = cap.freq
			else:
				if cap.freq <= freq_max:
					raise Exception("Out-of-order captures")
				else:
					freq_max = cap.freq
			overlap_bins = 0
			if freq_last is not None:
				if options.overlap is None:
					freq_diff = cap.freq - freq_last
					if freq_diff < 0:
						raise Exception("Negative frequency step")
					overlap = 1.0 - freq_diff / options.samp_rate
					#print "Overlap:", overlap
				else:
					overlap = options.overlap
				overlap_bins = options.fft_length * overlap
			print "[%05d] Freq: %s (%s - %s) overlap bins: %d" % (cnt, format_freq(cap.freq), format_freq(freq_bottom), format_freq(freq_top), overlap_bins)
			freq_line += [freq_bottom, freq_top]
			freq_center_line += [cap.freq]
			freq_last = cap.freq
			
			data = numpy.fromfile(cap.path, numpy.dtype(options.type))
			print "[%05d] Read %d items from %s" % (cnt, len(data), cap.path)
			
			_num_ffts, _fft_avg, _fft_min, _fft_max = calc_fft(data, options.fft_length, verbose=False)
			
			fft_avg = _stitch_array(fft_avg, _fft_avg, overlap_bins)
			fft_min = _stitch_array(fft_min, _fft_min, overlap_bins)
			fft_max = _stitch_array(fft_max, _fft_max, overlap_bins)
		
		print "Avg/min/max lengths: %d/%d/%d" % (len(fft_avg), len(fft_min), len(fft_max))
		freq_min -= (options.samp_rate / 2)
		freq_max += (options.samp_rate / 2)
		print "Freq range: %s - %s" % (format_freq(freq_min), format_freq(freq_max))
		
		x = numpy.linspace(freq_min, freq_max, len(fft_avg))
		
		fft_channel_graphs[channel_idx] = sub_graph = realtime_graph(
			parent=fft_graph,
			show=True,
			redraw=False,
			sub_title="Channel %i" % (channel_idx),
			pos=subplot_pos,
			y_limits=y_limits,
			x_range=options.fft_length,
			data=[fft_avg, fft_min, fft_max],
			x=x)
		
		for freq in freq_line:
			sub_graph.add_vert_line(freq, 'gray')
		for freq in freq_center_line:
			sub_graph.add_vert_line(freq)
		
		pos_count = pos_count + 1
	
	#fft_graph.redraw()
	fft_graph.go_modal()
	
	return 0

if __name__ == '__main__':
	main()
