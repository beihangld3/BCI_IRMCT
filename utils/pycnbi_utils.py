from __future__ import print_function
from __future__ import division

"""
PyCNBI utility functions

Note:
When exporting to Panda Dataframes format,
raw.as_data_frame() automatically scales data to volts unit by default.
This is the convention in MNE libararies.
Try raw.as_data_frame(scalings=dict(eeg=1.0, misc=1.0))

Kyuhwa Lee, 2014
Chair in Non-invasive Brain-machine Interface Lab (CNBI)
Swiss Federal Institute of Technology (EPFL)


This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""

import os, sys
import scipy.io
import mne
import numpy as np
import multiprocessing as mp
import q_common as qc
from pycnbi_config import CAP, LAPLACIAN
from scipy.signal import butter, lfilter,lfiltic, buttord

def slice_win(epochs_data, w_starts, w_length, psde, picks=None, epoch_id=None, flatten=True, verbose=False):
	'''
	Compute PSD values of a sliding window

	Params
		epochs_data: [channels] x [samples]
		w_starts: starting indices of sample segments
		w_length: window length in number of samples
		psde: MNE PSDEstimator object
		picks: subset of channels within epochs_data
		epochs_id: just to print out epoch ID associated with PID
		flatten: generate concatenated feature vectors
			If True: X = [windows] x [channels x freqs]
			If False: X = [windows] x [channels] x [freqs]

	Returns:
		[windows] x [channels*freqs] or [windows] x [channels] x [freqs]
	'''

	# raise error for wrong indexing
	def WrongIndexError(Exception):
		sys.stderr.write('\nERROR: %s\n'% Exception)
		sys.exit(-1)

	w_length= int(w_length)

	if epoch_id is None:
		print('[PID %d] Frames %d-%d'% (os.getpid(), w_starts[0], w_starts[-1] + w_length-1) )
	else:
		print('[PID %d] Epoch %d, Frames %d-%d'% (os.getpid(), epoch_id, w_starts[0], w_starts[-1] + w_length-1) )

	''' theoretical frequency resolution = sfreq / len(sample)
	if picks:
		X= np.empty( (0, len(picks) ) )
	else:
		X= np.empty( (0, epochs_data.shape[1]*PSD['tf_num'] ) ) # 2D: [overlapping_epochs] x [channels x freqs]
	'''
	X= None
	for n in w_starts:
		n= int(n)
		if n >= epochs_data.shape[1]:
			raise WrongIndexError('w_starts has an out-of-bounds index %d for epoch length %d.'% (n,epochs_data.shape[1]))
		window= epochs_data[:, n:(n + w_length)]

		# Try baselining
		#window= (window.T - np.mean( window, axis=1 )).T

		# dimension: psde.transform( [epochs x channels x times] )
		psd= psde.transform( window.reshape( (1, window.shape[0], window.shape[1]) ) )
		psd= psd.reshape( (psd.shape[0], psd.shape[1]*psd.shape[2]) )
		if picks:
			psd= psd[0][picks]
			psd= psd.reshape( (1, len(psd)) )

		if X is None:
			X= psd
		else:
			X= np.concatenate( (X, psd ), axis=0 )

		if verbose==True:
			print('[PID %d] processing frame %d / %d'% (os.getpid(), n, w_starts[-1]) )

	return X

def get_psd(epochs, psde, wlen, wstep, picks=None, flatten=True):
	"""
	Offline computation of multi-taper PSDs over sliding windows

	Params
	epochs: MNE Epochs object
	psde: MNE PSDEstimator object
	wlen: window length in frames
	wstep: window step in frames
	picks: channel picks
	flatten: boolean, see Returns section

	Returns
	-------
	if flatten==True:
		X_data: [epochs] x [windows] x [channels*freqs]
	else:
		X_data: [epochs] x [windows] x [channels] x [freqs]
	y_data: [epochs] x [windows]
	picks: feature indices to be used; use all if None

	TODO:
		Accept input as numpy array as well, in addition to Epochs object
	"""

	labels= epochs.events[:, -1]
	epochs_data= epochs.get_data()

	print('Opening pool of workers')
	pool= mp.Pool( mp.cpu_count() )

	# sliding window
	w_starts= np.arange(0, epochs_data.shape[2] - wlen, wstep)
	X_data= None
	y_data= None
	results= []
	for ep in np.arange(len(labels)):
		# for debugging (results not saved)
		#slice_win(epochs_data, w_starts, wlen, psde, picks, ep)

		# parallel psd computation
		results.append( pool.apply_async(slice_win, [epochs_data[ep], w_starts, wlen, psde, picks, ep]) )

	for ep in range(len(results)):
		r= results[ep].get() # windows x features
		X= r.reshape( (1, r.shape[0], r.shape[1]) ) # 1 x windows x features
		if X_data is None: X_data= X
		else: X_data= np.concatenate( (X_data, X), axis=0 )

		# speed comparison: http://stackoverflow.com/questions/5891410/numpy-array-initialization-fill-with-identical-values
		y= np.empty( (1, r.shape[0]) ) # 1 x windows
		y.fill( labels[ep] )
		if y_data is None: y_data= y
		else: y_data= np.concatenate( (y_data, y), axis=0 )
	pool.close()
	pool.join()

	if flatten:
		return X_data, y_data
	else:
		xs= X_data.shape
		nch= len(epochs.ch_names)
		return X_data.reshape(xs[0], xs[1], nch, int(xs[2]/nch)), y_data

# note that MNE already has find_events function
def find_events(events_raw):
	"""
	Find trigger values, rising from zero to non-zero
	"""
	events= [] # triggered event values other than zero

	# set epochs (frame start, frame end)
	ev_last= 0
	for et in range( len(events_raw) ):
		ev= events_raw[et]
		if ev != ev_last:
			if ev > 0:
				events.append( [ et, 0, ev ] )
			ev_last= ev

	return events

def find_event_channel(signals):
	"""
	Find event channel using heuristics for pcl files.

	Disclaimer: Not guaranteed to work.

	Input:
		signals: n_ch x times

	Output:
		channel index or None if not found.
	"""

	for c in range( signals.shape[0] ):
		if ( signals[c].astype(int)==signals[c] ).all() \
		and max(signals[c]) < 256 and min(signals[c])==0:
			return c
	return None

def raw2mat(infile, outfile):
	'''
	Convert raw data file to MATLAB file
	'''
	raw, events= load_raw(infile)
	header= dict(bads=raw.info['bads'], ch_names=raw.info['ch_names'],\
		sfreq=raw.info['sfreq'], events=events)
	scipy.io.savemat(outfile, dict(signals=raw._data, header=header) )
	print('\n>> Exported to %s'% outfile)

def pcl2mat_old(fpcl):
	"""
	For old format data only
	"""
	raw= qc.load_obj(fpcl)
	assert type(raw['signals'])==type(list())
	signals= np.array( raw['signals'][0] ) # samples x channels
	ts= raw['timestamps'][0]
	srate= raw['sample_rate']
	n_ch= raw['channels']
	if n_ch > 17: # BioSemi
		ev16= signals[:,0]-1 # first channel is event channel
		events_raw= 0xFF & ev16.astype(int) # keep only the low 8 bits
		events= find_events( events_raw )
	else:
		events= find_events( signals[:,-1] )

	print('Signal dimension:', signals.shape)
	print('Timestamp dimension:', len(ts) )
	print('Sampling rate:', srate)
	print('No. channels:', n_ch)
	data= dict(signals=signals, timestamps=ts, events=events, sample_rate=srate, n_channels=n_ch)
	fmat= fpcl[:-4] + '.mat'
	scipy.io.savemat( fmat, data )
	print('Saved data as', fmat)

def add_events_raw(rawfile, outfile, eventfile, overwrite=True):
	"""
	Add events from a file and save

	Note: If the event values already exists in raw file, the new event values
		will be added to the previous value instead of replacing them.
	"""

	raw= mne.io.Raw(rawfile, preload=True)
	events= mne.read_events(eventfile)
	raw.add_events( events, stim_channel='TRIGGER' )
	raw.save( outfile, overwrite=overwrite )

def export_morlet(epochs, filename):
	"""
	Export wavelet tranformation decomposition into Matlab format
	"""
	freqs= np.array( DWT['freqs'] ) # define frequencies of interest
	n_cycles= freqs / 2.  # different number of cycle per frequency
	power, itc= mne.time_frequency.tfr_morlet(epochs, freqs=freqs,
		n_cycles=n_cycles, use_fft=False, return_itc=True, n_jobs=mp.cpu_count() )
	scipy.io.savemat(filename, dict(power=power.data, itc=itc.data, freqs=freqs,\
		channels=epochs.ch_names, sfreq=epochs.info['sfreq'], onset=-epochs.tmin))

def event_timestamps_to_indices(sigfile, eventfile):
	"""
	Convert LSL timestamps to sample indices for separetely recorded events.

	Parameters:
	sigfile: raw signal file (Python Pickle) recorded with stream_recorder.py.
	eventfile: event file where events are indexed with LSL timestamps.

	Returns:
	events list, which can be used as an input to mne.io.RawArray.add_events().
	"""

	raw= qc.load_obj(sigfile)
	ts= raw['timestamps'].reshape(-1)
	ts_min= min(ts)
	ts_max= max(ts)
	events= []

	with open(eventfile) as f:
		for l in f:
			data= l.strip().split('\t')
			event_ts= float( data[0] )
			event_value= int( data[2] )
			# find the first index not smaller than ts
			next_index= np.searchsorted(ts, event_ts)
			if next_index >= len(ts):
				qc.print_c( '** WARNING: Event %d at time %.3f is out of time range (%.3f - %.3f).'% (event_value,event_ts,ts_min,ts_max), 'y' )
			else:
				events.append( [next_index, 0, event_value] )
			#print(events[-1])

	return events

def load_raw(rawfile, spfilter=None, spchannels=None, events_ext=None, multiplier=1):
	"""
	Loads data from a fif-format file.
	You can convert non-fif files (.eeg, .bdf, .gdf, .pcl) to fif format.

	Parameters:
	rawfile: (absolute) data file path
	spfilter: 'car' | 'laplacian' | None
	spchannels: None | list (for CAR) | dict (for LAPLACIAN)
		'car': channel indices used for CAR filtering. If None, use all channels except
			   the trigger channel (index 0).
		'laplacian': {channel:[neighbor1, neighbor2, ...], ...}
		*** Note ***
		Since PyCNBI puts trigger channel as index 0, data channel starts from index 1.
	events_mne: Add externally recorded events (e.g. software trigger).
				events_mne is of format: [ [sample_index1, 0, event_value1],... ]
	multiplier: Multiply all values except triggers (to convert unit).

	Returns:
	raw: mne.io.RawArray object. First channel (index 0) is always trigger channel.
	events: mne-compatible events numpy array object (N x [frame, 0, type])
	spfilter= {None | 'car' | 'laplacian'}

	"""

	if not os.path.exists(rawfile):
		qc.print_c('# ERROR: File %s not found'% rawfile, 'r')
		sys.exit(-1)

	extension= rawfile.split('.')[-1]
	assert extension in ['fif','fiff'], 'only fif format is supported'
	raw= mne.io.Raw(rawfile, preload=True)

	if 'TRIGGER' in raw.ch_names:
		tch_name= 'TRIGGER'
	elif 'STI ' in raw.ch_names: # e.g. 'STI 014'
		tch_name= 'STI '
	elif 'Trig1' in raw.ch_names:
		tch_name = 'Trig1'
	else:
		raise RuntimeError, 'No trigger channel found.'

	# find a value changing from zero to a non-zero value
	tch= raw.ch_names.index(tch_name)
	events= mne.find_events(raw, stim_channel=tch_name, shortest_event=1, uint_cast=True)

	n_channels= raw._data.shape[0]
	eeg_channels= list(range(n_channels))
	eeg_channels.pop(tch)
	if multiplier != 1:
		raw._data[eeg_channels] *= multiplier

	# apply spatial filter
	if spfilter=='car':
		if not spchannels:
			raw._data[eeg_channels]= raw._data[eeg_channels] - np.mean( raw._data[eeg_channels], axis=0 )
		else:
			raw._data[spchannels]= raw._data[spchannels] - np.mean( raw._data[spchannels], axis=0 )
	elif spfilter=='laplacian':
		if type(spchannels) is not dict:
			raise RuntimeError, 'For Lapcacian, SP_CHANNELS must be of a form {CHANNEL:[NEIGHBORS], ...}'
		rawcopy= raw._data.copy()
		for src in spchannels:
			nei= spchannels[src]
			raw._data[src]= rawcopy[src] - np.mean( rawcopy[nei], axis=0 )
	elif spfilter=='bipolar':
		raw._data[1:] -= raw._data[spchannels]
	elif spfilter is None:
		pass
	else:
		qc.print_c('# ERROR: Unknown spatial filter', spfilter, 'r')
		sys.exit(-1)

	return raw, events


def load_raw_old(rawfile, spfilter=None, spchannels=None, events_ext=None):
	"""
	** Deprecated function **
	Please use convert2fif to convert non-fif files to fif first.


	Returns raw data and events

	Supports gdf, bdf, fif, and Python raw format (pcl).
	Any non-fif file will be saved into .fif format in the fif/ directory after loading.

	Parameters:
	rawfile: (absolute) data file path
	spfilter: 'car' | 'laplacian' | None
	spchannels: None | list (for CAR) | dict (for LAPLACIAN)
		'car': channel indices used for CAR filtering. If None, use all channels except
			   the trigger channel (index 0).
		'laplacian': {channel:[neighbor1, neighbor2, ...], ...}
		*** Note ***
		Since PyCNBI puts trigger channel as index 0, data channel starts from index 1.
	events_mne: Add externally recorded events (e.g. software trigger).
				events_mne is of format: [ [sample_index1, 0, event_value1],... ]

	Returns:
	raw: mne.io.RawArray object. First channel (index 0) is always trigger channel.
	events: mne-compatible events numpy array object (N x [frame, 0, type])
	spfilter= {None | 'car' | 'laplacian'}

	"""

	if not os.path.exists(rawfile):
		qc.print_c('# ERROR: File %s not found'% rawfile, 'r')
		sys.exit(-1)

	rawfile= rawfile.replace('\\','/')
	dirs= rawfile.split('/')
	if len(dirs)==1: basedir= './'
	else: basedir= '/'.join(dirs[:-1]) + '/'
	extension= rawfile.split('.')[-1]
	basename= '.'.join( rawfile.split('.')[:-1] )
	raw= None
	events= []

	if extension == 'pcl':
		data= qc.load_obj(rawfile)

		if type(data['signals']) == list:
			print('Converting into numpy format')
			signals_raw= np.array(data['signals'][0]).T # to channels x samples
		else:
			signals_raw= data['signals'].T # to channels x samples
		sample_rate= data['sample_rate']
		events_raw= data['events']

		# BioSemi or gtec?
		if data['channels'] == 17:
			# move the trigger channel to the first row
			if find_event_channel( signals_raw ) != 16:
				qc.print_c('**** WARNING: Assuming GTEC_16 format. Double-check trigger channel !! *****', 'r')
			signals= np.concatenate( (signals_raw[16,:].reshape(1,-1), signals_raw[:16,:]) )
			info= mne.create_info( CAP['GTEC_16'], sample_rate, CAP['GTEC_16_INFO'] )
		elif data['channels'] >= 73:
			signals= signals_raw[:73,:] # trigger channel is already the first row
			sigtrig= signals[0,:] - 1
			signals[0,:]= 0xFF & sigtrig.astype(int) # keep only the low 8 bits
			info= mne.create_info( CAP['BIOSEMI_64'], sample_rate, CAP['BIOSEMI_64_INFO'] )
		elif data['channels'] == 24:
			qc.print_c('**** ASSUMING SmartBCI system with no trigger channel ****', 'y')
			if True:
				# A1=9, A2=16
				ear_avg= (signals_raw[8] + signals_raw[15]) / 2.0
				signals= signals_raw - ear_avg
				trigger= np.zeros( (1, signals_raw.shape[1]) )
				signals= np.vstack( (trigger, signals) )
			else:
				signals= signals_raw[:24,:] # trigger channel is already the first row
			sigtrig= signals[0,:]
			signals[0,:]= 0x00
			info= mne.create_info( CAP['SMARTBCI_24'], sample_rate, CAP['SMARTBCI_24_INFO'] )
		else: # ok, unknown format
			# guess trigger channel
			trig_ch= find_event_channel(signals_raw)
			if trig_ch is not None:
				qc.print_c('Found trigger channel %d. Moving to channel 0.'% trig_ch, 'y')
				signals= np.concatenate( (signals_raw[[trig_ch]], signals_raw[:trig_ch], signals_raw[trig_ch+1:]), axis=0 )
				assert signals_raw.shape == signals.shape
				num_eeg_channels= data['channels']-1
			else:
				# assuming no trigger channel exists, add a trigger channel to index 0 for consistency.
				qc.print_c('**** Unrecognized number of channels (%d). Adding an event channel to index 0.'% data['channels'], 'r')
				eventch= np.zeros( [1, signals_raw.shape[1]] )
				signals= np.concatenate( (eventch, signals_raw), axis=0 )
				num_eeg_channels= data['channels']

			ch_names= ['TRIGGER'] + ['CH%d'% (x+1) for x in range(num_eeg_channels)]
			ch_info= ['stim'] + ['eeg'] * num_eeg_channels
			info= mne.create_info( ch_names, sample_rate, ch_info )

	elif extension in ['fif','fiff']:
		raw= mne.io.Raw(rawfile, preload=True)

	elif extension in ['bdf','gdf']:
		# convert to mat using MATLAB (MNE's edf reader has an offset bug)
		matfile= basename+'.mat'
		if not os.path.exists(matfile):
			print('>> Converting input to mat file')
			run= "[sig,header]=sload('%s.%s'); save('%s.mat','sig','header');"%(basename,extension,basename)
			qc.matlab(run)
			if not os.path.exists(matfile):
				qc.print_c('>> ERROR: mat file convertion error.', 'r')
				sys.exit()

		mat= scipy.io.loadmat(matfile)
		os.remove( matfile )
		sample_rate= int( mat['header']['SampleRate'] )
		nch= mat['sig'].shape[1]

		if extension=='gdf':
			# Note: gdf might  have a software trigger channel
			if nch == 17:
				ch_names= CAP['GTEC_16']
				ch_info= CAP['GTEC_16_INFO'][:nch]
			else:
				ch_names= ['TRIGGER'] + ['ch%d'% x for x in range(1,nch)]
				ch_info= ['stim'] + ['eeg'] * (nch-1)

			# read events from header
			'''
			Important:
				event position may have the same frame number for two consecutive events
				It might be due to the CNBI software trigger bug
			Example:
				f1.20121220.102907.offline.mi.mi_rhlh.gdf (Two 10201's in evpos)
			'''
			evtype= mat['header']['EVENT'][0][0][0]['TYP'][0]
			evpos= mat['header']['EVENT'][0][0][0]['POS'][0]
			for e in range(evtype.shape[0]):
				label= int(evtype[e])
				events.append( [ int(evpos[e][0]), 0, label ] )

		elif extension=='bdf':
			# assume Biosemi always has the same number of channels
			if nch == 73:
				ch_names= CAP['BIOSEMI_64']
				extra_ch= nch - len(CAP['BIOSEMI_64_INFO'])
				extra_names= []
				for ch in range(extra_ch):
					extra_names.append( 'EXTRA%d'%ch )
				ch_names= ch_names + extra_names
				ch_info= CAP['BIOSEMI_64_INFO'] + ['misc'] * extra_ch
			else:
				qc.print_c('****** load_raw(): WARNING: Unrecognized number of channels (%d) ******'% nch, 'y')
				qc.print_c('The last channel will be assumed to be trigger. Press Enter to continue, or Ctrl+C to break.', 'r')
				raw_input()
				# Set the trigger to be channel 0 because later we will move it to channel 0.
				ch_names= ['TRIGGER'] + ['CH%d'% (x+1) for x in range(nch-1)]
				ch_info= ['stim'] + ['eeg'] * (nch-1)

		# Move the event channel to 0 (for consistency)
		signals_raw= mat['sig'].T # -> channels x samples
		signals= np.concatenate( (signals_raw[-1,:].reshape(1,-1), signals_raw[:-1,:]) )

		# Note: Biosig's sload() sometimes returns bogus event values so we use the following for events
		bdf= mne.io.read_raw_edf(rawfile, preload=True)
		events= mne.find_events(bdf)
		signals[-1][:]= bdf._data[-1][:] # overwrite with the correct event values

		info= mne.create_info( ch_names, sample_rate, ch_info )
	else:
		# unknown format
		qc.print_c('ERROR: Unrecognized file extension %s. It should be [.pcl | .fif | .fiff | .gdf | .bdf]'% extension, 'r')
		sys.exit(-1)

	if raw is None:
		# signals= channels x samples
		raw= mne.io.RawArray( signals, info )

		# check if software trigger
		trigch= raw.info['ch_names'].index('TRIGGER')
		if events != [] and max( raw[trigch][0][0] ) == 0:
			raw.add_events(events, stim_channel='TRIGGER')

		# external events with LSL timestamps
		if events_ext != None:
			if extension != 'pcl':
				qc.print_c('>> ERROR: external events can be only added to raw .pcl files', 'r')
				sys.exit(-1)
			events_index= event_timestamps_to_indices(rawfile, events_ext)
			raw.add_events( events_index, stim_channel='TRIGGER' )

		qc.make_dirs(basedir + 'fif/')
		fifname= basedir + 'fif/' + basename.split('/')[-1] + '.fif'
		raw.save(fifname, overwrite=True, verbose=False)
		print('Saving to', fifname)

	# find a value changing from zero to a non-zero value
	events= mne.find_events(raw, stim_channel='TRIGGER', shortest_event=1)

	# apply spatial filter
	n_channels= raw._data.shape[0]
	if spfilter=='car':
		if not spchannels:
			raw._data[1:]= raw._data[1:] - np.mean( raw._data[1:], axis=0 )
		else:
			raw._data[spchannels]= raw._data[spchannels] - np.mean( raw._data[spchannels], axis=0 )
	elif spfilter=='laplacian':
		if type(spchannels) is not dict:
			raise RuntimeError, 'For Lapcacian, SP_CHANNELS must be of a form {CHANNEL:[NEIGHBORS], ...}'
		rawcopy= raw._data.copy()
		for src in spchannels:
			nei= spchannels[src]
			raw._data[src]= rawcopy[src] - np.mean( rawcopy[nei], axis=0 )
	elif spfilter=='bipolar':
		raw._data[1:] -= raw._data[spchannels]
	elif spfilter is None:
		pass
	else:
		qc.print_c('# ERROR: Unknown spatial filter', spfilter, 'r')
		sys.exit(-1)

	return raw, events

def load_multi(flist, spfilter=None, spchannels=None, multiplier=1):
	"""
	Load multiple data files and concatenate them into a single series
	Event locations are updated accordingly with new offset

	Assumes the input data are of same type (same channels, sampling rate, etc.)
	See load_raw() for more details.

	"""

	if len(flist) == 0:
		raise RuntimeError, 'flist cannot be empty.'
	events= []
	signals= None
	for f in flist:
		raw, _= load_raw(f, spfilter=spfilter, spchannels=spchannels, multiplier=multiplier)
		if signals is None:
			signals= raw._data
		else:
			signals= np.concatenate( (signals, raw._data), axis=1 ) # append samples

	# create a concatenated raw object using the last raw object's info
	raws= mne.io.RawArray( signals, raw.info )

	# re-calculate event positions
	events= mne.find_events( raws, stim_channel='TRIGGER', shortest_event=1 )
# change back later....................................................................................................
#	events= mne.find_events( raws, stim_channel='Trig1', shortest_event=1 )

	return raws, events

#
#	Calculation of bandpass coefficients.
#	Order is computed automatically.
#	Note that if filter is unstable this function crashes (TODO handle problems)
#
def butter_bandpass(highcut, lowcut, fs, num_ch):
	low = lowcut/(0.5*fs)
	high = highcut/(0.5*fs)
	# get the order. TO BE DONE: Sometimes it fails
	ord = buttord(high, low, 2, 40)
	b, a = butter(2, [low, high], btype='band')
	zi = np.zeros([a.shape[0]-1, num_ch])
	return b, a, zi

def search_lsl(ignore_markers=False):
	import pylsl, time

	# look for LSL servers
	amp_list= []
	amp_list_backup= []
	while True:
		streamInfos= pylsl.resolve_streams()
		if len(streamInfos) > 0:
			for index, si in enumerate(streamInfos):
				amp_serial= pylsl.StreamInlet(si).info().desc().child('acquisition').child_value('serial_number').strip()
				amp_name= si.name()
				if 'Markers' in amp_name:
					amp_list_backup.append( (index, amp_name, amp_serial) )
				else:
					amp_list.append( (index, amp_name, amp_serial) )
			break
		print('No server available yet on the network...')
		time.sleep(1)

	if ignore_markers is False:
		amp_list += amp_list_backup

	qc.print_c('-- List of servers --', 'W')
	for i, (index, amp_name, amp_serial) in enumerate(amp_list):
		if amp_serial=='': amp_ser= 'N/A'
		else: amp_ser= amp_serial
		qc.print_c( '%d: %s (Serial %s)'% (i, amp_name, amp_ser), 'W' )

	if len(amp_list)==1:
		index= 0
		dong = 0
	else:
		dong = 1
		index= raw_input('Amp index? Hit enter without index to select the first server.\n>> ').strip()
		index = int(index)  # dong
		#amp_index, amp_name, amp_serial= [(0, u'g.USBamp-1',u'UB-2010.06.31')]
	#qc.shell()
	amp_index, amp_name, amp_serial= amp_list[index]
	#if dong == 1:
		#amp_index = 0
		#print('!!!!!!!!!!!!!!!')
	si= streamInfos[amp_index]
	assert amp_name == si.name()
	assert amp_serial == pylsl.StreamInlet(si).info().desc().child('acquisition').child_value('serial_number').strip()
	print('Selected %s (Serial: %s)'% (amp_name, amp_serial))

	return amp_name, amp_serial

def lsl_channel_list(inlet):
	"""
	Reads XML description of LSL header and returns channel list

	Input:
		pylsl.StreamInlet object
	Returns:
		ch_list: [ name1, name2, ... ]
	"""
	ch_list= []

	import xmltodict
	xml=inlet.info().as_xml()
	doc=xmltodict.parse(xml)
	channels= doc['info']['desc']['channels']['channel']
	for ch in channels:
		ch_list.append( ch['label'] )

	return ch_list
