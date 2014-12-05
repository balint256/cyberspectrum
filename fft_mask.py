#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  fft_mask.py
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

import interface
import numpy

from utils import *

import tcp_server

LISTEN_RETRY_INTERVAL = 5

class Mask():
	def __init__(self, name, options, spacing):
		self.name = name
		# options.mask_snr
		self.options = options
		self.spacing = (spacing + (spacing % 2)) / 2	# Compute either side
		self.points = None
	
	def build(self, fft_points):
		sorted_points = numpy.sort(fft_points)
		sorted_idices = numpy.argsort(fft_points)
		flags = [0] * len(fft_points)
		self.points = numpy.zeros(len(fft_points))
		for n in range(len(fft_points)):
			v = sorted_points[(len(sorted_points)-1) - n]
			i = sorted_idices[(len(sorted_points)-1) - n]
			if flags[i] != 0:
				continue
			self.points[i] = v
			if self.options.fixed_threshold is None:
				self.points[i] += self.options.mask_snr
			flags[i] = 1
			for j in range(self.spacing):
				lhs = i - j
				rhs = i + j
				if lhs >= 0:
					if flags[lhs] == 0:
						self.points[lhs] = self.points[i]
						flags[lhs] = 1
				if rhs < len(fft_points):
					if flags[rhs] == 0:
						self.points[rhs] = self.points[i]
						flags[rhs] = 1
	
	def check(self, fft_points, stitching=0):
		result = fft_points - self.points
		
		hits = []
		
		thru = False
		first = None
		for i in range(len(result)):
			if result[i] > 0.0:
				if not thru:
					if len(hits) > 0 and ((i - hits[-1][1]) <= stitching):
						#print "Stitching %d to %d" % (hits[-1][1], i)
						first = hits[-1][0]
						del hits[-1]
					else:
						first = i
					thru = True
			elif thru:
				if result[i] <= 0.0:
					hits += [(first, i)]
					thru = False
		
		if thru:
			hits += [(first, len(result)-1)]
		
		return hits

class EnergyDetector(interface.Module):
	def __init__(self, config, options, *args, **kwds):
		interface.Module.__init__(self, config, options, *args, **kwds)
		
		self.masks = []
		self.server = None
		
		#print "Initialised Energy Detector"
	
	def populate_options(self, parser):
		parser.add_option("--mask-snr", type="float", default=15.0, help="Mask threshold above noise floor (dB) [default: %default]")
		parser.add_option("--mask-space", type="float", default=150e3, help="Mask spacing (Hz) [default=%default]")
		parser.add_option("--mask-stitch", type="float", default=200e3, help="Mask spacing (Hz) [default=%default]")
		parser.add_option("--listen", type="int", default=10000, help="TCP server listen port [default=%default]")
		parser.add_option("--min-width", type="float", default=50e3, help="Minimum detection width (Hz) [default=%default]")
		parser.add_option("--fixed-threshold", type="float", default=None, help="Fixed detection threshold (dB) [default=%default]")
		parser.add_option("--only-save-hits", action="store_true", default=False, help="Only save data from acquisitions that trigger the detector [default=%default]")
	
	def init(self, usrp, info, states, state_machines, fft_graph, scope_graph):
		interface.Module.init(self, usrp, info, states, state_machines, fft_graph, scope_graph)
		
		if self.options.fixed_threshold is not None:
			print "Fixed mask threshold:", self.options.fixed_threshold
		
		for i in range(len(state_machines)):
			self.masks += [{}]
		
		print "Prepared %d mask lists" % (len(self.masks))
		
		self.server = tcp_server.ThreadedTCPServer(("", self.options.listen), silent=False)	# buffer_size=options.buffer_size, blocking_mode=options.blocking_send, send_limit=options.limit, 
		
		def _log_listen_retry(e, msg):
			print "    Socket error:", msg
			if (e == 98):
				print "    Waiting, then trying again..."
		
		self.server.start(retry=True, wait=LISTEN_RETRY_INTERVAL, log=_log_listen_retry)
		print "==> TCP server running in thread:", self.server.server_thread.getName()
	
	def shutdown(self):
		def _log_shutdown(client):
			print "==> Disconnecting client:", client.client_address
		
		self.server.shutdown(True, log=_log_shutdown)
	
	def start(self, count, current_hw_states):
		interface.Module.start(self, count, current_hw_states)
		
		self.triggered = []
	
	def query_fft(self, sample_idx, hw_state):
		return True
	
	def process(self, sample_idx, hw_state, s, fft_data, partial_name, fft_channel_graph, scope_channel_graph):
		self.server.send("[%04d] Processing chan #%d: %s\n" % (self.last_count, sample_idx, str(hw_state)))
		
		masks = self.masks[sample_idx]
		key = str(hw_state)
		first = False
		if key not in masks.keys():
			if self.options.fixed_threshold is not None:
				fft_points = [self.options.fixed_threshold]*len(fft_data['ave'])
			else:
				first = True
				# Use 'ave' and ignore spurs with options.min_width
				fft_points = fft_data['ave']	# 'ave', 'max'
				
				if False:
					window_length = 3	# MAGIC
					weights = numpy.repeat(1.0, window_length)/window_length
					ma_fft_points = numpy.convolve(fft_points, weights, 'valid')
					diff = len(fft_points) - len(ma_fft_points)
					ma_fft_points = numpy.append(numpy.array([float(ma_fft_points[0])]*(diff/2)), [ma_fft_points])
					ma_fft_points = numpy.append(ma_fft_points, [numpy.array([float(ma_fft_points[len(ma_fft_points)-1])]*(len(fft_points) - len(ma_fft_points)))])
					fft_points = ma_fft_points
			
			spacing = int((self.options.mask_space / self.config.rate) * len(fft_points))
			mask = Mask(key, self.options, spacing)
			mask.build(fft_points)
			masks[key] = mask
			
			built_msg = "Built mask for channel %d with key: %s" % (sample_idx, key)
			print built_msg
			self.server.send("[%04d] %s\n" % (self.last_count, built_msg))
		
		if not first:
			fft_points = fft_data['max']
			mask = masks[key]
		
		if fft_channel_graph:
			mask_points = []
			for i in range(len(fft_points)):
				f = hw_state.freq + ((self.config.rate / len(fft_points)) * (i - len(fft_points)/2 + 1))
				mask_points += [(f, mask.points[i])]
			
			fft_channel_graph.add_points(mask_points, 'bo')
		
		if first: return
		
		########################################################################
		# Process as normal
		
		stitching = int((self.options.mask_stitch / self.config.rate) * len(fft_points))
		
		hits = mask.check(fft_points, stitching)
		
		#print "Hits:", hits
		
		self.triggered += [(len(hits) > 0)]
		
		for hit in hits:
			f1 = hw_state.freq - self.config.rate/2
			f2 = self.config.rate / len(fft_points)
			hit_freq = [
				(f1 + f2*hit[0]),
				(f1 + f2*hit[1])
			]
			freq_range = hit_freq[1] - hit_freq[0]
			
			if freq_range < self.options.min_width:
				continue
			
			mid_point = hit_freq[0] + (freq_range / 2)
			points = fft_points[hit[0]:hit[1]]
			ave_power = numpy.average(points)
			
			hit_str = "Hit %04d-%04d: %s (%s wide) @ %f dBFs" % (hit[0], hit[1], format_freq(mid_point), format_freq(freq_range), ave_power)
			print hit_str
			
			def _log_send_error(client, e, msg):
				if e != 32: # Broken pipe
					print "==> While sending to", client.client_address, "-", e, msg
			
			self.server.send(hit_str + "\n", log=_log_send_error)
			
			if fft_channel_graph:
				hit_points = []
				for i in range(hit[0], hit[1]):
					f = hw_state.freq + ((self.config.rate / len(fft_points)) * (i - len(fft_points)/2 + 1))
					hit_points += [(f, fft_points[i])]
				fft_channel_graph.add_points(hit_points, 'ro')
		
	def query_save(self, which):
		if self.options.only_save_hits:
			if which == 'data':
				return self.triggered[-1]
			elif which == 'fft_graph':
				return reduce(lambda x,y: x or y, self.triggered)
		return None

def get_modules():
	return [{'class':EnergyDetector, 'name':"Energy Detector"}]

def main():
	return 0

if __name__ == '__main__':
	main()
