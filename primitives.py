#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  primitives.py
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

import math

from utils import *

class StateMachine():
	def __init__(self, states, idx=-1):
		self.states = states
		self.idx = idx
		self.loops = 0
	def next(self):
		self.idx += 1
		if self.idx == len(self.states):
			self.idx = 0
			self.loops += 1
		return self.current()
	def count(self):
		return len(self.states)
	def index(self):
		return self.idx
	def current(self):
		try:
			return self.states[self.idx]
		except Exception, e:
			print "Tried to access state #%d but there are only %d" % (self.idx+1, len(self.states))
			raise
	def loop_count(self):
		return self.loops

class Statistics():
	def __init__(self):
		self.reset()
	def reset(self):
		self._sum = 0.0
		self._min = None
		self._max = None
		self._count = 0
	def add(self, v):
		self._sum += float(v)
		if self._min is None:
			self._min = v
		else:
			self._min = min(self._min, v)
		if self._max is None:
			self._max = v
		else:
			self._max = max(self._max, v)
		self._count += 1
	def ave(self):
		return self._sum / self._count
	def count(self):
		return self._count
	def min(self):
		return self._min
	def max(self):
		return self._max

class ChannelCapabilites():
	def __init__(self, freq_range, antennas, gain_range):
		self.freq_range = freq_range
		self.antennas = antennas
		self.gain_range = gain_range
	def __str__(self):
		return "Freq: %s, Gains: %s, Antennas: %s" % (
			self.freq_range,
			self.gain_range,
			self.antennas
		)

class HardwareState():
	def __init__(self, state, gain, antenna, lo_offset, freq=None):
		self.state = state
		self.gain = gain
		self.antenna = antenna
		self.lo_offset = lo_offset
		self.freq = freq
	def get_antenna(self):
		if len(self.antenna) == 0:
			return "(default)"
		return self.antenna
	def __str__(self):
		res = "Gain: %f, Antenna: %s, LO offset: %s" % (self.gain, self.get_antenna(), format_freq(self.lo_offset))
		if self.freq is not None:
			res += ", Freq: %s" % (format_freq(self.freq))
		return res

class ScannerState():
	def __init__(self, freq_range, freq_config, channel, config, chan_caps):
		self.freq_range, self.freq_config, self.channel, self.config, self.chan_caps = freq_range, freq_config, channel, config, chan_caps
		# Other parameters filled in by 'setup' in FrequencyRange
		# FrequencyRange
		self.start, self.stop, self.step, self.edge = None, None, None, None
		# FrequencyConfig
		self.gains = None
		self.antennas = None
		self.lo_offset = None
	def __str__(self):
		return "Freq: %s-%s (%s steps, edge: %s), Gains: %s, Antennas: %s, LO offset: %f" % (
			format_freq(self.start), format_freq(self.stop), format_freq(self.step), self.edge,
			self.gains,
			self.antennas,
			self.lo_offset
		)
	def get_hw_states(self, calc_freqs):
		states = []
		
		if calc_freqs:
			if self.edge:
				start = self.start + (self.config.rate / 2.0)
				stop = self.stop - (self.config.rate / 2.0)
			else:
				start = self.start
				stop = self.stop
			
			if stop < start:
				raise Exception("Invalid frequency range")
			
			freq_range = stop - start
			steps = int(math.ceil(freq_range / self.step)) + 1
			#steps = max(1, int(math.ceil(freq_range / self.step)))	# Always have one step (when start & stop are the same)
			#if (freq_range > 0.0) and ((freq_range / self.step) == (freq_range // self.step)):	# If on boundary, include the last
			#	steps += 1
			
			if (freq_range > 0.0) and (freq_range < (self.config.rate / 2.0)):
				steps -= 1
			
			for i in range(steps):
				freq = start + (self.step * i)
				if freq > stop:
					freq = stop	# Should only happen once at the end if last is not equally spaced
				for gain in self.gains:
					for antenna in self.antennas:
						states += [HardwareState(self, gain, antenna, self.lo_offset, freq)]
		else:
			for gain in self.gains:
				for antenna in self.antennas:
					states += [HardwareState(self, gain, antenna, self.lo_offset)]
		
		return states

def _choose(val, default):
	if val is None:
		return default
	return val

# Sample rate will determine total available step
class FrequencyRange():
	def __init__(self, start=None, stop=None, step=None, edge=False):
		self.start = start	# None: lowest supported
		self.stop = stop	# None: highest supported
		self.step =	step	# Relative to sample rate (None: use default)
		self.edge = edge	# Start the LO at the range edge
	def setup(self, freq_config, channel, config, chan_caps):
		state = ScannerState(self, freq_config, channel, config, chan_caps)
		# FrequencyRange
		state.start = _choose(self.start, chan_caps.freq_range.start())
		state.stop = _choose(self.stop, chan_caps.freq_range.stop())
		state.step = _choose(self.step, freq_config.default_step) * config.rate
		state.edge = self.edge
		# FrequencyConfig
		if freq_config.gains is None:
			state.gains = channel.default_gains
			relative_gain = channel.relative_gain
		else:
			state.gains = freq_config.gains
			relative_gain = freq_config.relative_gain
		if not isinstance(state.gains, list):
			state.gains = [state.gains]
		# FIXME: Check gains in range
		if relative_gain:
			state.gains = [chan_caps.gain_range.start() + (chan_caps.gain_range.stop() - chan_caps.gain_range.start()) * g for g in state.gains]
		state.antennas = _choose(freq_config.antennas, channel.default_antennas)
		if isinstance(state.antennas, str):
			state.antennas = [state.antennas]
		elif state.antennas is None or state.antennas == False:
			state.antennas = ['']
		elif state.antennas == True:
			state.antennas = chan_caps.antennas
		# FIXME: Check antennas against caps
		state.lo_offset = channel.lo_offset
		return state

_default_frequency_ranges = [FrequencyRange()]

# Step is fraction of sample rate (bandwidth)
class FrequencyConfig():
	def __init__(self, frequency_ranges=_default_frequency_ranges, default_step=1.0, gains=None, relative_gain=None, antennas=None):
		self.frequency_ranges = frequency_ranges
		self.default_step = default_step
		self.gains = gains					# None: use default
		self.relative_gain = relative_gain	# None: use default
		self.antennas = antennas			# None: use default
	def setup(self, channel, config, chan_caps):
		freqs = []
		for fr in self.frequency_ranges:
			freqs += [fr.setup(self, channel, config, chan_caps)]
		return freqs

_default_frequency_config = [FrequencyConfig()]

# Maps to a side (daughterboard)
# Set defaults:
# Specify possible antennas to use (or all)
# gain, frequency, 
class ChannelConfig():
	def __init__(self, frequencies=_default_frequency_config, default_gains=[0.25], relative_gain=True, default_antennas=False, subdev="", lo_offset=0.0):
		self.frequencies = frequencies
		self.default_gains = default_gains
		self.relative_gain = relative_gain
		self.default_antennas = default_antennas	# False: default, True: all, or strings
		self.subdev = subdev
		self.lo_offset = lo_offset
	def setup(self, config, chan_caps):
		freqs = []
		for f in self.frequencies:
			freqs += f.setup(self, config, chan_caps)
		return freqs

class TunePolicy():
	def __init__(self, settling_time=0.0, consecutive_locks=1, wait_time=0.001, timeout=1, sensor_name="lo_locked"):
		self.settling_time = settling_time
		self.consecutive_locks = consecutive_locks
		self.wait_time = wait_time
		self.timeout = timeout
		self.sensor_name = sensor_name

_noise_floor = -130.0	# dB (lowest reasonable)
_default_sample_count = 1.0		# float: seconds, int: samples
_default_channel_config = [ChannelConfig()]

# Maps to a radio
# Specify sample rate
class Config():
	def __init__(self, name, length=_default_sample_count, args="", rate=1e6, channel_config=_default_channel_config, noise_floor=_noise_floor, linked=False, tune_policy=TunePolicy(), skip_samples=0):
		self.name = name
		self.length = length
		self.args = args
		self.rate = rate
		self.channel_config = channel_config
		self.linked = linked	# When channels are linked (single frequency, e.g. B210)
		self.noise_floor = noise_floor
		self.tune_policy = tune_policy
		self.skip_samples = skip_samples
		
		if isinstance(self.length, float):
			self.sample_count = int(self.rate * self.length)
		else:
			self.sample_count = self.length
		
		_max_sample_count = (1 << 31) - 1
		if (self.sample_count + self.skip_samples) > _max_sample_count:
			print "Clamping sample count from %d to %d (skipping %d)" % (self.sample_count, (_max_sample_count - skip_samples), skip_samples)
			self.sample_count = _max_sample_count - self.skip_samples
	def setup(self, chan_caps):
		if len(chan_caps) != len(self.channel_config):
			raise Exception("Number of channels in capabilities must match current hardware configuration")
		# FIXME: Check self.rate in supported rates
		channels = []
		idx = 0
		for cc in self.channel_config:
			channels += [cc.setup(self, chan_caps[idx])]
			idx += 1
		return channels

def main():
	return 0

if __name__ == '__main__':
	main()
