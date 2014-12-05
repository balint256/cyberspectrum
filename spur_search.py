#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  spur_search.py
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

import math

import numpy

import interface

def get_spurs(bins, freq_min, freq_max, snr=6.0, percent_noise_bins=80.0):
	"""
	Get a list of bins sticking out of the noise floor
	NOTE: This routine assumes flat noise floor with most bins as noise
	@param snr the number of db a bin needs to stick out of the noise floor
	@param percent_noise_bins is the minimum percentage of fft bins expected to be noise
	"""
	h = numpy.histogram(bins, numpy.arange(min(bins), max(bins), float(snr)/2.0))
	#print len(h[0]), h[0]
	#print len(h[1]), h[1]
	percent = 0.0
	for i in range(len(h[0])):
		percent += 100.0 * float(h[0][i])/float(len(h[0]))
		if percent > percent_noise_bins: break
	
	threshold = h[1][min(len(h[1])-1,i+2)]
	
	def _bin_to_freq(idx):
		freq_range = float(freq_max - freq_min)
		return idx * freq_range / (len(bins) - 1) + freq_min
	
	spurs = list()
	for i in range(len(bins)):
		if bins[i] > threshold: spurs.append((_bin_to_freq(i), bins[i]))

	return spurs 

class SpurSearch(interface.Module):
	def __init__(self, config, options, *args, **kwds):
		interface.Module.__init__(self, config, options, *args, **kwds)
		
		self.spur_log_file = None
		self.noise_log_file = None
		self.total_spur_count = 0
	
	def __del__(self):
		if self.spur_log_file: self.spur_log_file.close()
		if self.noise_log_file: self.noise_log_file.close()
	
	def populate_options(self, parser):
		parser.add_option("--spur-log", type="string", default=None, help="Spur log file [default=%default]")
		parser.add_option("--ignore-lo", action="store_true", help="Ignore LO spur", default=False)
		parser.add_option("--lo-tolerance", type="float", default=7.5e3, help="Ignore LO spur +/- from DC (Hz) [default: %default]")
		parser.add_option("--spur-snr", type="float", default=1.0, help="Spur threshold above noise floor (dB) [default: %default]")
		parser.add_option("--only-save-spurs", action="store_true", default=False, help="Only save image when spurs are detected [default: %default]")
		parser.add_option("--noise-log", type="string", default=None, help="Noise floor log file [default=%default]")
	
	def init(self, usrp, info, states, state_machines, fft_graph, scope_graph):
		interface.Module.init(self, usrp, info, states, state_machines, fft_graph, scope_graph)
		
		if not self.spur_log_file and self.options.spur_log is not None and len(self.options.spur_log) > 0:
			self.spur_log_file = open(self.options.spur_log, "w")
		if not self.noise_log_file and self.options.noise_log is not None and len(self.options.noise_log) > 0:
			self.noise_log_file = open(self.options.noise_log, "w")
	
	def start(self, count, current_hw_states):
		interface.Module.start(self, count, current_hw_states)
		
		self.total_spur_count = 0
	
	def query_stop(self, channel_idx, state_machine, hw_state):
		return (state_machine.loops > 0)
	
	def query_fft(self, sample_idx, hw_state):
		return True
	
	def process(self, sample_idx, hw_state, s, fft_data, partial_name, fft_channel_graph, scope_channel_graph):
		spurs_detected = []
		lo_spurs = []
		noise = None
		freq_min = hw_state.freq - self.config.rate/2
		freq_max = hw_state.freq + self.config.rate/2
		
		fft_avg = fft_data['ave']
		
		hz_per_bin = math.ceil(self.config.rate / len(fft_avg))
		lo_bins = int(math.ceil(self.options.lo_tolerance / hz_per_bin))
		#print "Skipping %i LO bins" % (lo_bins)
		lhs = fft_avg[0:((len(fft_avg) + 1)/2) - ((lo_bins-1)/2)]
		rhs = fft_avg[len(fft_avg)/2 + ((lo_bins-1)/2):]
		#print len(fft_avg), len(lhs), len(rhs)
		fft_minus_lo = numpy.concatenate((lhs, rhs))
		#noise = numpy.average(numpy.array(fft_minus_lo))
		noise = 10.0 * math.log10(numpy.average(10.0 ** (fft_minus_lo / 10.0))) # dB
		print ("\t[%i] Noise (skipped %i LO FFT bins)" % (sample_idx, lo_bins)), noise, "dB"
		
		lo_freq = hw_state.freq + hw_state.lo_offset
		fig_name = "fft-%s.png" % (partial_name)	# Same as scanner.py
		
		if self.noise_log_file:
			self.noise_log_file.write("%d,%d,%f,%f,%f,%s,%f,%s\n" % (
				self.last_count,
				sample_idx,
				hw_state.freq,
				lo_freq,
				hw_state.gain,
				hw_state.get_antenna(),
				noise,
				fig_name,
			))
		
		spurs = get_spurs(fft_avg, freq_min, freq_max)	# snr=6.0, percent_noise_bins=80.0
		
		spur_threshold = noise + self.options.spur_snr
		
		for spur_freq, spur_level in spurs:
			if spur_level > spur_threshold:
				if self.options.ignore_lo and abs(lo_freq - spur_freq) < self.options.lo_tolerance:
					#print "\t[%i]\tLO   @ %f MHz (%03f dBm) for LO %f MHz (offset %f Hz)" % (channel, spur_freq, spur_level, lo_freq, (spur_freq-lo_freq))
					lo_spurs += [(spur_freq, spur_level)]
				else:
					spurs_detected += [(spur_freq, spur_level)]
					#d = {
					#	'id': id,
					#	'spur_level': spur_level,
					#	'spur_freq': spur_freq,
					#	'lo_freq': lo_freq,
					#	'channel': channel,
					#	'noise_floor': noise,
					#}
					#print '\t\tSpur:', d
					print "\t[%i]\tSpur @ %f Hz (%03f dBFS) for LO %f MHz (offset %f Hz)" % (
						sample_idx,
						spur_freq,
						spur_level,
						lo_freq,
						(spur_freq-lo_freq)
					)
					if self.spur_log_file:
						self.spur_log_file.write("%d,%d,%f,%f,%f,%s\n" % (
							self.last_count,
							sample_idx,
							spur_freq,
							spur_level,
							lo_freq,
							fig_name,
						))
					self.total_spur_count += 1
		
		if fft_channel_graph is not None:
			fft_channel_graph.add_points(spurs_detected)
			fft_channel_graph.add_horz_line(noise, 'gray', '--', id='noise')
			fft_channel_graph.add_horz_line(spur_threshold, 'gray', '-', id='spur_threshold')
			fft_channel_graph.add_points(lo_spurs, 'go')
	
	def query_save(self, which):
		if which == 'fft_graph':
			if self.options.only_save_spurs:
				return (self.total_spur_count > 0)
		return None
	
	def shutdown(self):
		return

def get_modules():
	return [{'class':SpurSearch, 'name':"Spur Search"}]

def main():
	return 0

if __name__ == '__main__':
	main()
