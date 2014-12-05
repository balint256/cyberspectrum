#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  scanner.py
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

import sys, socket, traceback, time, datetime, os, signal
from optparse import OptionParser

import numpy

try:
	from realtime_graph import *
except Exception, e:
	print "Failed to import realtime_graph:", e
	print "Clone https://github.com/balint256/baz-utils and add lib/python to your PYTHONPATH"
	realtime_graph = None
	matplotlib = None

try:
	from fft_tools import *
except Exception, e:
	print "Failed to import fft_tools:", e
	print "Clone https://github.com/balint256/baz-utils and add lib/python to your PYTHONPATH"
	raise e

try:
	import wx	# To detect when MPL window is closed
except Exception, e:
	print "Failed to import wx:", e
	wx = Exception()
	wx._core = Exception()
	wx._core.PyDeadObjectError = Exception()

from gnuradio import gr, uhd
try:
	from baz import usrp_acquire
except Exception, e:
	print "baz.usrp_acquire will not be available (failed to import):", e
	usrp_acquire = None

from utils import *
from primitives import *

_configs = None

try:
	import config as _config
	_configs = _config._config
except Exception, e:
	print "Could not import configuration:", e
	print "Make sure you have a config.py (e.g. make a copy of config.model.py)"
	#traceback.print_exc()
	raise e

running = True

def signal_term_handler(signal, frame):
	global running
	print "Received SIGTERM"
	running = False

# RuntimeError: EnvironmentError: IOError: Radio ctrl (A) packet parse error - AssertionError: packet_info.packet_count == (seq_to_ack & 0xfff)

########################################################################
# Magic to turn pointers into numpy arrays
# http://docs.scipy.org/doc/numpy/reference/arrays.interface.html
########################################################################
def pointer_to_ndarray(addr, dtype, nitems, read_only=False):
	class array_like:
		__array_interface__ = {
			'data' : (int(addr), read_only),
			'typestr' : dtype.base.str,
			'descr' : dtype.base.descr,
			'shape' : (nitems,) + dtype.shape,
			'strides' : None,
			'version' : 3
		}
	return numpy.asarray(array_like()).view(dtype.base)

class RestartException(Exception):
	pass

def main():
	parser = OptionParser(usage="%prog: [options] [config name] [-- module-specific options]")
	
	parser.add_option("-l", "--length", type="string", default=None, help="Override capture length [default=%default]")
	parser.add_option("-a", "--args", type="string", default=None, help="Override UHD device args [default=%default]")
	#parser.add_option("-f", "--fifo", type="string", default="", help="GPS FIFO path [default=%default]")
	#parser.add_option("-p", "--port", type="int", default=12345, help="GPS server port [default=%default]")
	parser.add_option("-G", "--graph", action="store_true", default=False, help="Graph samples [default=%default]")
	parser.add_option("-L", "--location", type="string", default=None, help="Capture location [default=%default]")
	parser.add_option("-s", "--skip", type="int", default=0, help="Iterations to skip [default=%default]")
	parser.add_option("--no-gps", action="store_true", default=False, help="Don't query GPS [default=%default]")
	parser.add_option("--wait", action="store_true", default=False, help="Wait after each iteration [default: %default]")
	parser.add_option("", "--fft-length", type="int", default=2048, help="FFT length [default=%default]")
	parser.add_option("", "--scope-length", type="int", default=2048, help="Scope length [default=%default]")
	parser.add_option("-m", "--modules", type="string", default="", help="Load additional modules [default=%default]")
	parser.add_option("-v", "--verbose", action="store_true", default=False, help="Verbose output [default: %default]")
	parser.add_option("-P", "--pad-fft", action="store_true", default=False, help="Pad FFTs [default: %default]")
	parser.add_option("-S", "--fft-step", type="int", default=1, help="FFT step size [default=%default]")
	parser.add_option("-w", "--window", type="string", default="hamming", help="FFT window function [default=%default]")
	parser.add_option("", "--abort", action="store_true", default=False, help="Abort on error that is otherwise retried [default: %default]")
	parser.add_option("", "--restart", action="store_true", default=False, help="Restart on error that is otherwise retried [default: %default]")
	parser.add_option("--once", action="store_true", default=False, help="Exit instead of looping [default=%default]")
	
	(options, args) = parser.parse_args()
	
	config = None
	if len(args) >= 1 and args[0] != "-":
		for c in _configs:
			if c.name.lower() == args[0].lower():
				config = c
		if config == None and len(args[0]) > 0:
			print "Config '%s' not found" % (args[0])
			return
	
	if config == None:
		config = _config.Config("(default)")
	
	print "Using config:", config.name
	
	if options.graph and realtime_graph is None:
		print "Cannot graph when realtime_graph unavailable"
		return
	
	if options.args is not None:
		print "Overriding args \"%s\" with \"%s\"" % (options.args, options.args)
		config.args = options.args
	
	if options.length is not None and len(options.length) > 0:
		print "Overriding length '%s' with '%s'" % (str(config.length), options.length)
		config.length = float(options.length)
		if not options.length.find('.'):
			config.length = int(config.length)
	
	window_fn = None
	if len(options.window) > 0 and options.window != "-":
		try:
			window_fn = getattr(numpy, options.window)
			print "Using window function:", options.window
		except:
			print "Window function not found:", options.window
			raise
	else:
		print "Not using a window function"
	
	module_list = []
	
	if len(options.modules) > 0:
		module_names = options.modules.split(',')
		for module_name in module_names:
			exec("import " + module_name)
			print "Loaded:", module_name
			module = sys.modules[module_name]
			module_list += module.get_modules()
	
	if gr.enable_realtime_scheduling() != gr.RT_OK:
		print "Failed to enable realtime scheduling - do you have sufficient permissions?"
	
	nmea_sensors = ["gps_gpgga", "gps_gprmc"]
	
	channel_count = len(config.channel_config)
	print "Sampling from %d channels" % (channel_count)
	channels = range(channel_count)
	
	signal.signal(signal.SIGTERM, signal_term_handler)
	print "Installed signal handler"
	
	fft_graph = None
	fft_channel_graphs = {}
	scope_graph = None
	scope_channel_graphs = {}
	
	font = {
		#'family' : 'normal',
		#'weight' : 'bold',
		'size'   : 10
	}
	
	if options.graph and matplotlib is not None:
		matplotlib.rc('font', **font)
	
	global running
	while running:
		modules = []
		module_options = OptionParser()
		
		for m in module_list:
			module_instance = m['class'](config, options)
			module_instance.populate_options(module_options)
			modules += [module_instance]
			print "Added:", m['class']
		
		(module_opts, extra_args) = module_options.parse_args(args=[])	# Init defaults
		
		if len(args) > 1:	# First will be config name
			(module_opts, extra_args) = module_options.parse_args(args=args[1:], values=module_opts)
		
		for opt in module_options.option_list:
			if opt.dest is None:
				continue
			o = getattr(module_opts, opt.dest)
			#print opt.dest, "=", o
			setattr(options, opt.dest, o)
		
		stream_args = uhd.stream_args(
			cpu_format="fc32",	# Fixed for finite_acquisition
			channels=channels,
		)
		
		# FIXME: This can throw on X310
		try:
			usrp = uhd.usrp_source(
				device_addr=config.args,
				stream_args=stream_args,
			)
		except RuntimeError, e:
			print "Likely UHD exception:", e
			print "Waiting..."
			time.sleep(5)
			print "Trying again..."
			continue
		except Exception, e:
			print "Unknown exception:", e
			break
		
		usrp_acquire_src = None
		if usrp_acquire:
			#usrp_acquire_src = usrp_acquire(usrp.get_device(), stream_args)
			usrp_acquire_src = usrp_acquire.make_from_source(usrp.to_basic_block(), stream_args)
			print "Using usrp_acquire"
		else:
			print "Using uhd"
		
		info = {}
		uhd_info = usrp.get_usrp_info()
		for k in uhd_info.keys():
			#print k, "=", info.get(k)
			info[k] = uhd_info.get(k)
		print "Device info:", info
		
		mboard_sensor_names = usrp.get_mboard_sensor_names()
		_available_nmea_sensors = []
		for sensor_name in mboard_sensor_names:
			if sensor_name in nmea_sensors:
				_available_nmea_sensors += [sensor_name]
		
		if len(_available_nmea_sensors) == 0:
			print "GPS not available"
		else:
			print "GPS sensors available:", _available_nmea_sensors
		
		usrp.set_samp_rate(config.rate)
		
		print "Sample rate:", usrp.get_samp_rate()
		#print "Sample rates:", len(usrp.get_samp_rates())
		#print "Center freq:", usrp.get_center_freq()
		#print "Freq range:", usrp.get_freq_range()
		#print "Gain:", usrp.get_gain()
		#print "Gain names:", usrp.get_gain_names()
		#print "Gain range:", usrp.get_gain_range()
		#print "Antenna:", usrp.get_antenna()
		#print "Antennas:", usrp.get_antennas()
		
		chan_caps = []
		chan_sensors = []
		for chan_idx in channels:
			chan_caps += [_config.ChannelCapabilites(usrp.get_freq_range(chan_idx), usrp.get_antennas(chan_idx), usrp.get_gain_range(chan_idx))]
			chan_sensors += [usrp.get_sensor_names(chan_idx)]
		print "Capabilites:"
		print "\n".join(map(str, chan_caps))
		print "Sensors:"
		print "\n".join(map(str, chan_sensors))
		
		mboard = 0
		
		# FIXME: Args
		#usrp.set_clock_source(ref, mboard)
		#usrp.set_time_source(pps, mboard)
		
		print "Clock source:", usrp.get_clock_source(mboard)
		print "Time source: ", usrp.get_time_source(mboard)
		print "Clock rate:  ", usrp.get_clock_rate()
		print "Time now:    ", usrp.get_time_now().get_real_secs()
		print "Time PPS:    ", usrp.get_time_last_pps().get_real_secs()
		
		subdev_spec = " ".join([cc.subdev for cc in config.channel_config]).strip()
		# FIXME: Validation
		
		if len(subdev_spec) > 0:
			usrp.set_subdev_spec(subdev_spec)
		
		print "Subdev spec:", usrp.get_subdev_spec()
		
		# HW channels must map directly to order in ChannelConfig
		states = config.setup(chan_caps)
		
		print "States:"
		for i in range(len(states)):
			print "%d:" % (i)
			print "\n".join(map(str, states[i]))
		
		# Init state machine
		state_machines = []
		for state in states:
			hw_state = []
			for s in state:
				hw_state += s.get_hw_states(True)
			state_machines += [StateMachine(hw_state)]
		
		################################
		
		gui_fft_length = options.fft_length
		gui_scope_length = options.scope_length
		
		# FIXME: CTRL+C handling doesn't work with this
		if options.graph:
			if fft_graph:
				fft_graph.close()
			if scope_graph:
				scope_graph.close()
			
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
			scope_graph = realtime_graph(title="Scope", show=True, manual=True, redraw=False, figsize=figsize, padding=padding)
			
			pos_count = 0
			y_limits = (config.noise_floor - 10.0, -30*0)	# For FFT	# FIXME: Arg
			for channel_idx in channels:
				#if channel_count > 2:
				#    pos_offset = ((pos_count % 2) * 2) + (pos_count / 2) + 1   # Re-order column-major
				#else:
				pos_offset = pos_count + 1
				subplot_pos = (channel_pos + pos_offset)
				
				fft_channel_graphs[channel_idx] = sub_graph = realtime_graph(parent=fft_graph, show=True, redraw=False, sub_title="Channel %i" % (channel_idx), pos=subplot_pos, y_limits=y_limits, x_range=gui_fft_length)
				sub_graph.add_horz_line(config.noise_floor)
				
				scope_channel_graphs[channel_idx] = sub_graph = realtime_graph(parent=scope_graph, show=True, redraw=False, sub_title="Channel %i" % (channel_idx), pos=subplot_pos)	#, x_range=NUM_BINS_SPUR, y_limits=y_limits
				
				pos_count = pos_count + 1
			
			# So the GUIs are updated
			fft_graph.redraw()
			scope_graph.redraw()
		
		for m in modules: m.init(usrp, info, states, state_machines, fft_graph, scope_graph)
		
		################################
		
		count = 0
		#tune_times = []
		tune_stats = Statistics()
		iteration_stats = Statistics()
		acquisition_stats = Statistics()
		computation_stats = Statistics()
		skip = options.skip
		
		while running:
			count += 1
			idx = count - 1
			
			iteration_start = time.time()
			
			current_hw_states = []
			
			for channel_idx in channels:
				state_machine = state_machines[channel_idx]
				hw_state = state_machine.next()
				if options.once and state_machine.loops > 0:	# This will catch the *first* state machine loop
					running = False
					break
				
				for m in modules:
					if m.query_stop(channel_idx, state_machine, hw_state):
						running = False
						break
				if not running:
					break
				
				current_hw_states += [hw_state]
			
			print
			#print "Current HW states:"
			#print "\n".join(map(str, current_hw_states))
			
			if not running:
				break
			
			if skip > 0:
				skip -= 1
				continue
			
			try:
				if count > 0 and options.wait:
					print "Waiting: ",
					raw_input()
				
				#print
				print "Iteration:", count
				
				for channel_idx in channels:
					state_machine = state_machines[channel_idx]
					print "Channel #%d state machine index: %03d/%03d" % (channel_idx, (state_machine.index()+1), state_machine.count())
				
				#ts = time.time()
				#time_str = time.strftime("%a, %d %b %Y %H:%M:%S", ts)
				time_now = datetime.datetime.now()
				time_now_str = time_now.strftime("%Y/%m/%d %H:%M:%S.%f")
				print "Host time:", time_now_str
				print "USRP time:", usrp.get_time_now().get_real_secs()
				
				if not options.no_gps:
					for sensor_name in _available_nmea_sensors:
						sensor_value = usrp.get_mboard_sensor(sensor_name)
						value = sensor_value.value.strip()
						if value == "":
							continue
						print value
				
				for m in modules: m.start(count, current_hw_states)	# FIXME: GPS info?
				
				for channel_idx in channels:	# FIXME: Any callback into modules here? E.g. modify state/skip?
					hw_state = current_hw_states[channel_idx]
					
					print "Chan %d: %s" % (channel_idx, str(hw_state))
					
					if not (hw_state.antenna is None or len(hw_state.antenna) == 0):
						usrp.set_antenna(hw_state.antenna, channel_idx)
					
					# This here prevents LO offset from being used
					#if config.linked and channel_idx > 0:
					#	continue
					
					# [Anything after this should be first channel, or in unlinked front-end]
					
					usrp.set_gain(hw_state.gain, channel_idx)	# Still set the gain first, in case moving to a band of powerful signals
					
					tune_start = time.time()
					usrp.set_center_freq(uhd.tune_request(hw_state.freq, hw_state.lo_offset), channel_idx)
					tune_duration = time.time() - tune_start
					#tune_times += [tune_duration]
					tune_stats.add(tune_duration)
					print "Tune time: %f ms (average: %f ms, min: %f ms, max: %f ms)" % (
						tune_duration*1e3,
						tune_stats.ave()*1e3,	#numpy.average(tune_times)*1e3,
						tune_stats.min()*1e3,	#min(tune_times)*1e3,
						tune_stats.max()*1e3)	#max(tune_times)*1e3)
					
					if config.linked and channel_idx > 0:
						continue
					
					if config.tune_policy.settling_time > 0:
						time.sleep(config.tune_policy.settling_time)
					
					channel_sensor_names = chan_sensors[channel_idx]
					if config.tune_policy.sensor_name in channel_sensor_names:
						consecutive_locks = 0
						time_start = time.time()
						sys.stdout.write("Waiting for LO lock: ")
						sys.stdout.flush()
						while (time.time() - time_start) < config.tune_policy.timeout:
							lo_locked_sensor = usrp.get_sensor(config.tune_policy.sensor_name)
							lo_locked = lo_locked_sensor.to_bool()
							if lo_locked:
								sys.stdout.write("*")
								sys.stdout.flush()
								
								consecutive_locks += 1
								if consecutive_locks == config.tune_policy.consecutive_locks:
									break
							else:
								sys.stdout.write("_")
								sys.stdout.flush()
								
								if consecutive_locks > 0:
									print "Re-trying tune..."
									usrp.set_center_freq(uhd.tune_request(hw_state.freq, hw_state.lo_offset), channel_idx)	# Try again
								
								consecutive_locks = 0
							time.sleep(config.tune_policy.wait_time)
						
						print
						
						if consecutive_locks != config.tune_policy.consecutive_locks:
							print "Failed to lock!"
							continue
				
				retry = True
				total_sample_count = config.sample_count + config.skip_samples
				while retry:
					retry = False	# Default path is to break from the loop
					
					acquisition_start = time.time()
					
					if usrp_acquire_src:
						stream_now = (len(channels) == 1)
						delay = 0.01
						timeout = 1.0
						sample_ptrs = usrp_acquire_src.finite_acquisition_v(total_sample_count, stream_now=stream_now, delay=delay, skip=config.skip_samples, timeout=timeout)	# FIXME: skip samples (offset into array)
						samples = []
						acquired_sample_count = sample_ptrs[-1]
						for sample_ptr_idx in range(len(sample_ptrs)-1):
							samples += [pointer_to_ndarray(sample_ptrs[sample_ptr_idx], numpy.dtype(numpy.complex64), acquired_sample_count, True)]
					else:
						samples = usrp.finite_acquisition_v(total_sample_count)
					
					acquisition_duration = time.time() - acquisition_start
					acquisition_stats.add(acquisition_duration)
					print "Acquisition time: %f ms (average: %f ms, min: %f ms, max: %f ms)" % (
						acquisition_duration*1e3,
						acquisition_stats.ave()*1e3,
						acquisition_stats.min()*1e3,
						acquisition_stats.max()*1e3)
					
					computation_state = time.time()
					
					partial_name = "%s-%s-%05d-%d-%d-%.1f-%s" % (config.name, time_now_str, count, channel_count, int(hw_state.freq), hw_state.gain, hw_state.get_antenna())
					partial_name = partial_name.replace("/", "_").replace(":", "_").replace(" ", "_")
					
					expected_sample_count = config.sample_count
					if usrp_acquire_src is None:
						expected_sample_count += config.skip_samples
					
					sample_idx = 0
					for s in samples:
						if len(s) == 0:
							if options.abort:
								print "Channel %d: didn't receive any samples - aborting." % (sample_idx)
								running = False
								break
							elif options.restart:
								print "Channel %d: didn't receive any samples - restarting." % (sample_idx)
								raise RestartException()
							print "Channel %d: didn't receive any samples - retrying..." % (sample_idx)
							retry = True
							break
						elif len(s) != expected_sample_count:
							
							if options.abort:
								print "Channel %d: only received %d samples (%d short) - aborting." % (sample_idx, len(s), (expected_sample_count-len(s)))
								running = False
								break
							elif options.restart:
								print "Channel %d: only received %d samples (%d short) - restarting." % (sample_idx, len(s), (expected_sample_count-len(s)))
								raise RestartException()
							print "Channel %d: only received %d samples (%d short) - retrying..." % (sample_idx, len(s), (expected_sample_count-len(s)))
							retry = True
							break
						print "Channel %d: received %d samples" % (sample_idx, len(s))
						
						if config.skip_samples > 0 and usrp_acquire_src is None:
							print "Removing skipped samples..."
							s = numpy.array(s[config.skip_samples:])	# This is slow
							print "Removed skipped samples."
						
						hw_state = current_hw_states[sample_idx]
						
						title = "Chan %d: %s" % (sample_idx, hw_state)
						
						fft_length = gui_fft_length
						
						force_fft = False
						for m in modules:
							if m.query_fft(sample_idx, hw_state):
								force_fft = True
						
						num_ffts = 0
						fft_avg, fft_min, fft_max = None, None, None
						if fft_graph is not None or force_fft:
							num_ffts, fft_avg, fft_min, fft_max = calc_fft(
								s,
								gui_fft_length,
								verbose=options.verbose,
								pad=options.pad_fft,
								step=options.fft_step,
								window=window_fn
							)
						
						freq_min = hw_state.freq-config.rate/2
						freq_max = hw_state.freq+config.rate/2
						
						fft_channel_graph = None
						if fft_graph is not None:
							fft_channel_graph = fft_channel_graphs[sample_idx]
							#freqs = [fft_result.bin_to_freq(bin) for bin in range(len(fft_dbm))]
							freqs = numpy.linspace(freq_min, freq_max, len(fft_avg))
							fft_channel_graph.update(data=[fft_avg, fft_min, fft_max], sub_title=title, redraw=False, x=freqs, points=[])	#, points=spurs_detected
						
						scope_channel_graph = None
						if scope_graph is not None:
							scope_channel_graph = scope_channel_graphs[sample_idx]
							mag_samples = numpy.absolute(numpy.array(s[:gui_scope_length]))
							scope_channel_graph.update(data=mag_samples, sub_title=title, redraw=False)
						
						for m in modules: m.process(sample_idx, hw_state, s, {'num':num_ffts, 'ave':fft_avg, 'min':fft_min, 'max':fft_max}, partial_name, fft_channel_graph, scope_channel_graph)
						
						if options.location is not None:
							force_save = None
							for m in modules:
								query_result = m.query_save('data')
								if query_result == True:
									force_save = True
									break	# Prioritise True
								elif query_result == False:
									force_save = False
							
							if force_save is None or force_save == True:	# Default to save
								capture_file_name = "%s-%d.cfile" % (partial_name, sample_idx)
								capture_file_path = os.path.join(options.location, capture_file_name)
								print "Saving to:", capture_file_path
								try:
									#f = open(capture_file_path, "w")
									s.astype('c8').tofile(capture_file_path)
									#f.close()
								except Exception, e:
									print "Failed to save samples to file:", capture_file_path
									print e
						
						sample_idx += 1
					
					for m in modules: m.stop((sample_idx == channel_count))
					
					computation_duration = time.time() - computation_state
					computation_stats.add(computation_duration)
					print "Computation time: %f ms (average: %f ms, min: %f ms, max: %f ms)" % (
						computation_duration*1e3,
						computation_stats.ave()*1e3,
						computation_stats.min()*1e3,
						computation_stats.max()*1e3)
					
					if sample_idx == channel_count:	# If it didn't break out of the inner-loop prematurely...
						if fft_graph is not None:
							fft_graph.redraw()
							
							if options.location is not None:
								force_save = None
								for m in modules:
									query_result = m.query_save('fft_graph')
									if query_result == True:
										force_save = True
										break	# Prioritise True
									elif query_result == False:
										force_save = False
								
								if force_save is None or force_save == True:	# Default to save
									fig_name = "fft-%s.png" % (partial_name)
									fig_name = os.path.join(options.location, fig_name)
									print "Saving FFT graph to: \"%s\"" % (fig_name)
									fft_graph.save(fig_name)
						if scope_graph is not None:
							scope_graph.redraw()
				
				# Done
				
				iteration_end = time.time()
				iteration_duration = iteration_end - iteration_start
				iteration_stats.add(iteration_duration)
				print "Iteration time: %f ms (average: %f ms, min: %f ms, max: %f ms)" % (
						iteration_duration*1e3,
						iteration_stats.ave()*1e3,
						iteration_stats.min()*1e3,
						iteration_stats.max()*1e3)
			except RestartException:
				print "Deleting USRP for restart..."
				del usrp_acquire_src
				del usrp
				break
			except wx._core.PyDeadObjectError:
				print "GUI window closed"
				running = False
				break
			except KeyboardInterrupt:
				print "Stopping..."
				running = False
				break
			except RuntimeError, e:
				print "Likely UHD runtime exception:", e
				#print "Args:", e.args
				#print "Message:", e.message
				
				# Abort if too many errors in short space of time
				# How to handle such errors as bad MCR? Try dummy iteration?
				
				print "Deleting USRP..."
				del usrp_acquire_src
				del usrp
				
				if options.abort:
					running = False
				else:
					print
				
				break
			except IOError, e:
				print "Caught an I/O error: %s" % (e)
				traceback.print_exc()
				running = False
				break
			except Exception, e:
				print "Caught unhandled exception (%s): %s" % (type(e), e)
				traceback.print_exc()
				running = False
				break
		
		for m in modules: m.shutdown()

if __name__ == '__main__':
	main()
