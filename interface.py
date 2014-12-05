#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  interface.py
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

class Module():
	def __init__(self, config, options, *args, **kwds):
		self.config = config
		self.options = options
	def populate_options(self, parser):
		pass
	def init(self, usrp, info, states, state_machines, fft_graph, scope_graph):
		self.usrp = usrp
		self.info = info
		self.states = states
		self.state_machines = state_machines
		self.fft_graph = fft_graph
		self.scope_graph = scope_graph
	def start(self, count, current_hw_states):
		self.last_count = count
		self.last_hw_states = current_hw_states
	def query_stop(self, channel_idx, state_machine, hw_state):
		return False
	def query_fft(self, sample_idx, hw_state):
		return False
	def process(self, sample_idx, hw_state, s, fft_data, partial_name, fft_channel_graph, scope_channel_graph):
		return
	def stop(self, successful):
		return
	def query_save(self, which):	# data, fft_graph
		return None
	def shutdown(self):
		return

def main():
	return 0

if __name__ == '__main__':
	main()
