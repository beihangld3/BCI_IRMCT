# transfer .fit files to .mat
from __future__ import print_function
from __future__ import division

# set the data path
dataPath = r'D:\data\Records\fif'

import os, sys, scipy, scipy.io
import mne

def get_file_list(dataPath, fullPath=False, recursive=False):
	dataPath.replace('\\', '/')
	if not dataPath[-1]=='/': dataPath+='/'

	if recursive==False:
		if fullPath==True:
			filelist = [dataPath+f for f in os.listdir(dataPath) if os.path.isfile(dataPath+'/'+f) and f[0]!='.']
		else:
			filelist = [f for f in os.listdir(dataPath) if os.path.isfile(dataPath+'/'+f) and f[0]!='.']
	else:
		filelist = []
		for root, dirs, files in os.walk(dataPath):
			root = root.replace('\\', '/')
			if fullPath == True:
				[filelist.append(root+'/'+f) for f in files]
			else:
				[filelist.append(f) for f in files]
	return sorted(filelist)

def load_raw(rawfile):
	if not os.path.exists(rawfile):
		print('Error: file %s not found'% rawfile)
		sys.exit(-1)
	extensions = rawfile.split('.')[-1]
	assert extensions in ['fif', 'fiff'], 'only fif format is supported'
	raw = mne.io.Raw(rawfile, preload = True)
	if 'TRIGGER' in raw.ch_names:
		tch_name = 'TRIGGER'
	else:
		raise RuntimeError, 'No trigger channel found, opps, pls check usb2lpt'
	events = mne.find_events(raw, stim_channel=tch_name, shortest_event = 1, uint_cast = True)
	return raw, events
	
# to-do list:
# print error msg in color text
#def print_c

# spatial filter

if __name__=='__main__':
	for rawfile in get_file_list(dataPath, fullPath=True, recursive=True):
		if rawfile[-4:] != '.fif': continue
		raw, events = load_raw(rawfile)
		fs = raw.info['sfreq'] # sample frequency
		data = dict(signals = raw._data, events = events, fs = fs)
		matfile = '.'.join(rawfile.split('.')[:-1]) + '.mat'
		scipy.io.savemat(matfile, data)
		print('\nExported to %s' % matfile)
	print('\nDone.')


