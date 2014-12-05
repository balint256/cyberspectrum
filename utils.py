#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  utils.py
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

def format_freq(f, decimals=None, units=True):
	unit = ''
	if f >= 1e9:
		f /= 1e9
		unit = 'G'
	elif f >= 1e6:
		f /= 1e6
		unit = 'M'
	elif f >= 1e3:
		f /= 1e3
		unit = 'k'
	if decimals is None:
		fmt = "%f"
	else:
		fmt = "%%f.%d" % (decimals)
	freq_str = fmt % f
	if units:
		freq_str += " %sHz" % (unit)
	return freq_str

def main():
	return 0

if __name__ == '__main__':
	main()
