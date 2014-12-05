#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  config.model.py
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

from primitives import *	# See in here for more info

_rate = 25e6
_sample_count = 0.5	# float: seconds, int: # samples

_b200_frequency_config = [FrequencyConfig(default_step=0.85, frequency_ranges=[FrequencyRange(5e9, 6e9)])]

_test_channel_config_b210 = ChannelConfig(default_gains=[25],relative_gain=False,default_antennas='TX/RX',frequencies=_b200_frequency_config,lo_offset=0)
_test_channel_config_b210_lo_offset = ChannelConfig(default_gains=[25],relative_gain=False,default_antennas='TX/RX',frequencies=_b200_frequency_config,lo_offset=15e6)
_b2x0_channel_config = [_test_channel_config_b210]	# Single channel
_b210_channel_config = [_test_channel_config_b210, _test_channel_config_b210]	# Dual channel (remember to set 'master_clock_rate' lower than 30.72e6 wherever this is used!)

_x300_frequency_config = [FrequencyConfig(default_step=0.85)]

# These are created separately as they'll be filled with different ranges automatically for a dual-channel motherboard populated with a WBX & SBX
_x300_channel_config_wbx = ChannelConfig(default_gains=[15],relative_gain=False,default_antennas='TX/RX',frequencies=_x300_frequency_config,lo_offset=15e6)
_x300_channel_config_sbx = ChannelConfig(default_gains=[15],relative_gain=False,default_antennas='TX/RX',frequencies=_x300_frequency_config,lo_offset=15e6)

_x300_tune_policy = TunePolicy(400e-6, 5)

# 'linked' indicates the radio has a shared LO for multiple channels

_config = [
	Config("x300", _sample_count, "type=x300", _rate/2, channel_config=[_x300_channel_config_wbx,_x300_channel_config_sbx], tune_policy=_x300_tune_policy, skip_samples=256),
	
	Config("b200",   _sample_count, "type=b200,master_clock_rate=25e6,num_recv_frames=512", _rate/2, channel_config=_b2x0_channel_config, linked=True, skip_samples=256),
	Config("b200-1", _sample_count, "type=b200,num_recv_frames=512", 16e6, channel_config=_b2x0_channel_config, linked=True, skip_samples=256),
	Config("b210", _sample_count, "type=b200,num_recv_frames=512,master_clock_rate=16e6", 16e6, channel_config=_b2x0_channel_config*2, linked=True, skip_samples=256),	# Cheating with "*2" to create two channels
	
	Config("b200-spur", 2500000, "type=b200,master_clock_rate=25e6,num_recv_frames=512", 25e6,
		channel_config=[ChannelConfig(default_gains=[25],relative_gain=False,default_antennas='TX/RX',
			frequencies=[FrequencyConfig(default_step=0.85, frequency_ranges=[FrequencyRange(50e6, 6e9)])]	# In-line frequencies
		)],
		linked=True, skip_samples=256),
	
	Config("e300", _sample_count, "", 4e6, channel_config=_b2x0_channel_config, linked=True, skip_samples=256),
]

def main():
	for c in _config:
		print c.name

if __name__ == '__main__':
	main()
