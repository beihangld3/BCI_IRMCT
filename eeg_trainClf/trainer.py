from __future__ import print_function
from __future__ import division

"""
trainer.py

Train a classifier using Linear Discriminant Analysis or Random Forests

Sections
--------
EVENTS: Dictionary containing pairs of {event_label:event_value}.
DATA: A directory containing raw files.
CHANNELS: Channels to consider.
EPOCHS: Epoch range in seconds relative to events.
FILTERS: Spatial and spectral filters.
FEATURES: Underlynig feature represenation.


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

# start
import os, sys, timeit, platform
import numpy as np
import matplotlib.pyplot as plt
import mne, mne.io
import multiprocessing as mp
import pdb
import scipy
import sklearn.metrics as skmetrics
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.cross_validation import StratifiedShuffleSplit, LeaveOneOut
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import cross_val_score
from mne import Epochs, pick_types
import traceback

import pycnbi_config
import q_common as qc
import pycnbi_utils as pu
from rlda import rLDA

def get_psd_feature(epochs_train, window, psdparam, feat_picks=None):
	"""
	params:
	  epochs_train: mne.Epochs object or list of mne.Epochs object.
	  window: time window range for computing PSD. Can be [a,b] or [ [a1,b1], [a1,b2], ...]
	"""

	if type(window[0]) is list:
		sfreq= epochs_train[0].info['sfreq']
		wlen= []
		w_frames= []
		# multiple PSD estimators, defined for each epoch
		if type(psdparam) is list:
			print('MULTIPLE PSD FUNCTION NOT IMPLEMENTED YET.')
			sys.exit(-1)

			'''
			TODO: implement multi-PSD for each epoch
			'''
			assert len(psdparam)==len(window)
			for i, p in enumerate(psdparam):
				if p['wlen']==None:
					wl= window[i][1] - window[i][0]
				else:
					wl= p['wlen']
				wlen.append( wl )
				w_frames.append( int(sfreq * wl) )
		# same PSD estimator for all epochs
		else:
			for i, e in enumerate(window):
				if psdparam['wlen']==None:
					wl= window[i][1] - window[i][0]
				else:
					wl= psdparam['wlen']
				assert wl > 0
				wlen.append( wl )
				w_frames.append( int(sfreq * wl) )
	else:
		sfreq= epochs_train.info['sfreq']
		wlen= window[1] - window[0]
		if psdparam['wlen'] is None:
			psdparam['wlen']= wlen
		w_frames= int(sfreq * psdparam['wlen']) # window length

	psde= mne.decoding.PSDEstimator(sfreq=sfreq, fmin=psdparam['fmin'],\
		fmax=psdparam['fmax'], bandwidth=None, adaptive=False, low_bias=True,\
		n_jobs=1, normalization='length', verbose=None)

	print('\n>> Computing PSD for training set')
	if type(epochs_train) is list:
		X_all= []
		for i, ep in enumerate(epochs_train):
			X, Y_data= pu.get_psd(ep, psde, w_frames[i], psdparam['wstep'], feat_picks)
			X_all.append(X)
		# concatenate along the feature dimension
		X_data= np.concatenate( X_all, axis=2 )
	else:
		X_data, Y_data= pu.get_psd(epochs_train, psde, w_frames, psdparam['wstep'], feat_picks)

	# return a class-like data structure
	return dict(X_data= X_data, Y_data= Y_data, wlen= wlen, w_frames= w_frames, psde= psde)

def get_timelags(epochs, wlen, wstep, downsample=1, picks=None):
	"""
	Get concatenated timelag features

	Parameters
	----------
	epochs: input signals
	wlen: window length (# time points) in downsampled data
	wstep: window step in downsampled data
	downsample: downsample signal to be 1/downsample length
	picks: ignored for now

	Returns
	-------
	X: [epochs] x [windows] x [channels*freqs]
	y: [epochs] x [labels]
	"""

	wlen= int(wlen)
	wstep= int(wstep)
	downsample= int(downsample)
	X_data= None
	y_data= None
	labels= epochs.events[:, -1] # every epoch must have event id
	epochs_data= epochs.get_data()
	n_channels= epochs_data.shape[1]
	# trim to the nearest divisible length
	epoch_ds_len= int(epochs_data.shape[2] / downsample)
	epoch_len= downsample * epoch_ds_len
	range_epochs= np.arange(epochs_data.shape[0])
	range_channels= np.arange(epochs_data.shape[1])
	range_windows= np.arange(epoch_ds_len-wlen, 0, -wstep)
	X_data= np.zeros( (len(range_epochs), len(range_windows), wlen*n_channels ) )

	# for each epoch
	for ep in range_epochs:
		epoch= epochs_data[ep, :, :epoch_len]
		ds= qc.average_every_n( epoch.reshape(-1), downsample ) # flatten to 1-D, then downsample
		epoch_ds= ds.reshape( n_channels, -1 ) # recover structure to channel x samples
		# for each window over all channels
		for i in range(len(range_windows)):
			w= range_windows[i]
			X= epoch_ds[:, w:w+wlen].reshape(1,-1) # our feature vector
			X_data[ep, i, :]= X

		# fill labels
		y= np.empty( (1, len(range_windows) ) ) # 1 x windows
		y.fill( labels[ep] )
		if y_data is None: y_data= y
		else: y_data= np.concatenate( (y_data, y), axis=0 )

	return X_data, y_data

def feature2chz(x, fqlist, picks, ch_names=None):
	"""
	Label channel, frequency pair for PSD feature indices

	Input
	------
	x: feature index
	picks: channels used (channel 0 being trigger channel)
	fqlist: list of frequency bands
	ch_names: list of complete channel names

	Output
	-------
	(channel, frequency)

	"""

	n_fq= len(fqlist)
	hz= fqlist[ x % n_fq ]
	ch= int( x / n_fq ) # 0-based indexing
	ch_names= np.array(ch_names)
	try:
		if ch_names is not None:
			return ch_names[picks[ch]], hz
		else:
			return picks[ch], hz
	except:
		traceback.print_exc()
		raise RuntimeError, '\n**** Error in feature2chz(). ****'

def balance_samples(X, Y, balance_type, verbose=False):
	if balance_type=='OVER':
		"""
		Oversample from classes that lack samples
		"""
		label_set= np.unique(Y)
		max_set= []
		X_balanced= np.array(X)
		Y_balanced= np.array(Y)

		# find a class with maximum number of samples
		for c in label_set:
			yl= np.where( Y == c )[0]
			if len(max_set)==0 or len(yl) > max_set[1]:
				max_set= [c, len(yl)]

		for c in label_set:
			if c == max_set[0]: continue
			yl= np.where( Y == c )[0]
			extra_samples= max_set[1] - len(yl)
			extra_idx= np.random.choice( yl, extra_samples )
			X_balanced= np.append(X_balanced, X[extra_idx], axis=0)
			Y_balanced= np.append(Y_balanced, Y[extra_idx], axis=0)

	elif balance_type=='UNDER':
		"""
		Undersample from classes that are excessive
		"""
		label_set= np.unique(Y)
		min_set= []

		# find a class with minimum number of samples
		for c in label_set:
			yl= np.where( Y == c )[0]
			if len(min_set)==0 or len(yl) < min_set[1]:
				min_set= [c, len(yl)]

		yl= np.where( Y == min_set[0] )[0]
		X_balanced= np.array(X[yl])
		Y_balanced= np.array(Y[yl])

		for c in label_set:
			if c == min_set[0]: continue
			yl= np.where( Y == c )[0]
			reduced_idx= np.random.choice( yl, min_set[1] )
			X_balanced= np.append(X_balanced, X[reduced_idx], axis=0)
			Y_balanced= np.append(Y_balanced, Y[reduced_idx], axis=0)
	else:
		print('>> ERROR: Unknown balancing type', balance_type)
		sys.exit(-1)

	if verbose == True:
		print('\n>> Number of trials BEFORE balancing')
		for c in label_set:
			print( '%s: %d'%(cfg.tdef.by_value[c], len(np.where( Y == c )[0])) )
		print('\n>> Number of trials AFTER balancing')
		for c in label_set:
			print( '%s: %d'%(cfg.tdef.by_value[c], len(np.where( Y_balanced == c )[0])) )

	return X_balanced, Y_balanced

def crossval_epochs(cv, epochs_data, labels, cls, label_names=None, do_balance=False):
	"""
	Epoch (trial) based cross-validation

	cv: scikit-learn cross-validation object
	epochs_data: np.array of [epochs x samples x features]
	cls: classifier
	labels: vector of integer labels
	label_names: associated label names {0:'Left', 1:'Right', ...}
	do_balance: oversample or undersample to match the number of samples among classes
	"""

	scores= []
	cnum= 1
	timer= qc.Timer()
	label_set= np.unique(labels)
	num_labels= len(label_set)
	cm= np.zeros( (num_labels, num_labels) )
	if label_names==None:
		label_names= {l:'%s'%l for l in label_set}

	# select train and test trial ID's
	for train, test in cv:
		timer.reset()
		X_train= np.concatenate( epochs_data[train] )
		X_test= np.concatenate( epochs_data[test] )
		Y_train= np.concatenate( labels[train] )
		Y_test= np.concatenate( labels[test] )
		if do_balance != False:
			X_train, Y_train= balance_samples(X_train, Y_train, do_balance)
			X_test, Y_test= balance_samples(X_test, Y_test, do_balance)

		cls.n_jobs= mp.cpu_count()
		cls.fit( X_train, Y_train )
		cls.n_jobs= 1
		#score= cls.score( X_test, Y_test )
		Y_pred= cls.predict( X_test )
		score= skmetrics.accuracy_score(Y_test, Y_pred)
		cm += skmetrics.confusion_matrix(Y_test, Y_pred, label_set)
		scores.append( score )
		print('Cross-validation %d / %d (%.2f) - %.1f sec'% (cnum, len(cv), score,timer.sec()) )
		cnum += 1

	# show confusion matrix
	cm_rate= cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
	print('\nY: ground-truth, X: predicted')
	for l in label_set:
		print('%-5s'% label_names[l][:5], end='\t')
	print()
	for r in cm_rate:
		for c in r:
			print('%-5.2f'% c, end='\t')
		print()
	print('Average accuracy: %.2f'% np.mean(scores) )

	'''
	# plot confusion matrix
	plt.matshow(cm_rate)
	plt.title('Confusion matrix')
	plt.colorbar()
	plt.ylabel('True label')
	plt.xlabel('Predicted label')
	plt.show()
	'''

	return np.array(scores)

def cva_features(datadir):
	for fin in qc.get_file_list(datadir, fullpath=True):
		if fin[-4:] != '.gdf': continue
		fout= fin+'.cva'
		if os.path.exists(fout):
			print('Skipping', fout)
			continue
		print("cva_features('%s')"% fin)
		qc.matlab("cva_features('%s')"% fin)

def run_trainer(cfg, ftrain, interactive=False):
	# feature selection?
	datadir= cfg.DATADIR
	feat_picks= None
	txt= 'all'
	if cfg.USE_CVA:
		fcva= ftrain[0] + '.cva'
		if os.path.exists(fcva):
			feat_picks= open(fcva).readline().strip().split(',')
			feat_picks= [int(x) for x in feat_picks]
			print('\n>> Using only selected features')
			print(feat_picks)
			txt= 'cva'

	if hasattr(cfg, 'BALANCE_SAMPLES'):
		do_balance= cfg.BALANCE_SAMPLES
	else:
		do_balance= False

	# preprocessing, epoching and PSD computation
	n_epochs= {}
	if cfg.LOAD_PSD:
		raise RunetimeError, 'SORRY, CODE NOT FINISHED.'
		labels= np.array( [] )
		X_data= None
		Y_data= None
		sfreq= None
		ts= None
		te= None
		for fpsd in qc.get_file_list(datadir, fullpath=True):
			if fpsd[-4:] != '.psd': continue
			data= qc.load_obj(fpsd)
			labels= np.hstack( (labels,data['Y'][:,0]) )
			if X_data is None:
				sfreq= data['sfreq']
				tmin= data['tmin']
				tmax= data['tmax']
				'''
				TODO: implement multi-segment epochs
				'''
				if type(cfg.EPOCH[0]) is list:
					print('MULTI-SEGMENT EPOCH IS NOT SUPPORTED YET.')
					sys.exit(-1)
				if cfg.EPOCH[0] < tmin or cfg.EPOCH[1] > tmax:
					raise RuntimeError, '\n*** Epoch time range is out of data range.'
				ts= int( (cfg.EPOCH[0] - tmin) * sfreq / data['wstep'] )
				te= int( (cfg.EPOCH[1] - tmin) * sfreq / data['wstep'] )

				# X: trials x channels x features
				X_data= data['X'][:, ts:te, :]
				Y_data= data['Y'][:, ts:te]
			else:
				X_data= np.vstack( (X_data, data['X'][:, ts:te, :] ) )
				Y_data= np.vstack( (Y_data, data['Y'][:, ts:te] ) )
		assert( len(labels) > 0 )
		psde= data['psde']
		psd_tmin= data['tmin']
		psd_tmax= data['tmax']
		picks= data['picks']
		w_frames= int(sfreq * data['wlen']) # window length
		psdparams= dict(fmin=data['fmin'], fmax=data['fmax'], wlen=data['wlen'], wstep=data['wstep'])

		if 'classes' in data:
			triggers= data['classes']
		else:
			triggers= { c:cfg.tdef.by_value[c] for c in set(labels) }

		spfilter= data['spfilter']
		spchannels= data['spchannels']
		tpfilter= data['tpfilter']
		for ev in data['classes']:
			n_epochs[ev]= len( np.where( Y_data[:,0] == data['classes'][ev] )[0] )

	else:
		spfilter= cfg.SP_FILTER
		tpfilter= cfg.TP_FILTER

		# Load multiple files
		if hasattr(cfg,'MULTIPLIER'):
			multiplier= cfg.MULTIPLIER
		else:
			multiplier= 1
		raw, events= pu.load_multi(ftrain, spfilter=spfilter, multiplier=multiplier)
		if cfg.LOAD_EVENTS_FILE is not None:
			events= mne.read_events(cfg.LOAD_EVENTS_FILE)

		triggers= { cfg.tdef.by_value[c]:c for c in set(cfg.TRIGGER_DEF) }

		# Pick channels
		if cfg.CHANNEL_PICKS is None:
			picks= pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False, exclude='bads')
		else:
			picks= []
			for c in cfg.CHANNEL_PICKS:
				if type(c)==int:
					picks.append(c)
				elif type(c)==str:
					picks.append( raw.ch_names.index(c) )
				else:
					raise RuntimeError, 'CHANNEL_PICKS is unknown format.\nCHANNEL_PICKS=%s'% cfg.CHANNEL_PICKS

		if max(picks) > len(raw.info['ch_names']):
			print('ERROR: "picks" has a channel index %d while there are only %d channels.'%\
				( max(picks),len(raw.info['ch_names']) ) )
			sys.exit(-1)

		# Spatial filter
		if cfg.SP_CHANNELS is None:
			spchannels= pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False, exclude='bads')
		else:
			spchannels= []
			for c in cfg.SP_CHANNELS:
				if type(c)==int:
					spchannels.append(c)
				elif type(c)==str:
					spchannels.append( raw.ch_names.index(c) )
				else:
					raise RuntimeError, 'SP_CHANNELS is unknown format.\nSP_CHANNELS=%s'% cfg.SP_CHANNELS

		# Spectral filter
		if tpfilter is not None:
			raw= raw.filter( tpfilter[0], tpfilter[1], picks=picks, n_jobs= mp.cpu_count() )
		if cfg.NOTCH_FILTER is not None:
			raw= raw.notch_filter( cfg.NOTCH_FILTER, picks=picks, n_jobs= mp.cpu_count() )

		# Read epochs
		try:
			if type( cfg.EPOCH[0] ) is list:
				epochs_train= []
				for ep in cfg.EPOCH:
					epochs_train.append( Epochs(raw, events, triggers, tmin=ep[0], tmax=ep[1], proj=False,\
						picks=picks, baseline=None, preload=True, add_eeg_ref=False, verbose=False, detrend=None) )
			else:
				epochs_train= Epochs(raw, events, triggers, tmin=cfg.EPOCH[0], tmax=cfg.EPOCH[1], proj=False,\
					picks=picks, baseline=None, preload=True, add_eeg_ref=False, verbose=False, detrend=None)
		except:
			print('\n*** (trainer.py) ERROR OCCURRED WHILE EPOCHING ***\n')
			traceback.print_exc()
			if interactive:
				print('Dropping into a shell.\n')
				pdb.set_trace()
			raise RuntimeError

		label_set= np.unique(triggers.values())
		sfreq= raw.info['sfreq']

		# Compute features
		if cfg.FEATURES=='PSD':
			res= get_psd_feature(epochs_train, cfg.EPOCH, cfg.PSD, feat_picks)
			X_data= res['X_data']
			Y_data= res['Y_data']
			wlen= res['wlen']
			w_frames= res['w_frames']
			psde= res['psde']
			psdfile= '%s/psd/psd-train.pcl'% datadir


		elif cfg.FEATURES=='TIMELAG':
			'''
			TODO: Implement multiple epochs for timelag feature
			'''
			if type(epcohs_train) is list:
				print('MULTIPLE EPOCHS NOT IMPLEMENTED YET FOR TIMELAG FEATURE.')
				sys.exit(-1)

			X_data, Y_data= get_timelags(epochs_train, cfg.TIMELAG['w_frames'], cfg.TIMELAG['wstep'], cfg.TIMELAG['downsample'])
		elif cfg.FEATURES=='WAVELET':
			'''
			TODO: Implement multiple epochs for wavelet feature
			'''
			if type(epcohs_train) is list:
				print('MULTIPLE EPOCHS NOT IMPLEMENTED YET FOR WAVELET FEATURE.')
				sys.exit(-1)

			############################### DO WE NEED SLIDING WINDOW ?????????
			X_data, Y_data= None, None
			for ev in epochs_train.event_id:
				e= 0
				for ep in epochs_train[ev]:
					e += 1
					freqs= np.arange(4, 30, 2)
					n_cycles= freqs / 2
					tfr= mne.time_frequency.cwt_morlet( ep, sfreq, freqs=freqs, n_cycles=n_cycles )
					tlen= 0.8
					tfr= np.log( np.abs( tfr[:,:,round(-sfreq*tlen):] ) )
					'''
					qc.make_dirs('%s/mat'% cfg.DATADIR)
					scipy.io.savemat('%s/mat/tfr-%s-%d.mat'% (cfg.DATADIR,ev,e), {'tfr':tfr[2]})
					'''
					feat= tfr.reshape(1,-1)
					if X_data is None:
						X_data= feat
					else:
						X_data= np.concatenate( (X_data, feat), axis=0 )
				# Y_data dimension is different here !
				y= np.empty( (epochs_train[ev]._data.shape[0] ) ) # windows x 1
				y.fill( epochs_train.event_id[ev] )
				if Y_data is None:
					Y_data= y
				else:
					Y_data= np.concatenate( (Y_data, y) )

			cls= RandomForestClassifier(n_estimators=cfg.RF['trees'], max_features='auto',\
				max_depth=cfg.RF['maxdepth'], n_jobs=mp.cpu_count() )#, class_weight={cfg.tdef.LOS:20, cfg.tdef.LO:1})
			scores= []
			cnum= 1
			timer= qc.Timer()
			num_labels= len(label_set)
			cm= np.zeros( (num_labels, num_labels) )

			# select train and test trial ID's
			from sklearn import cross_validation
			cv= cross_validation.ShuffleSplit( X_data.shape[0], n_iter=20, test_size=0.1 )
			for train, test in cv:
				timer.reset()
				X_train= X_data[train]
				X_test= X_data[test]
				Y_train= Y_data[train]
				Y_test= Y_data[test]
				if do_balance != False:
					X_train, Y_train= balance_samples(X_train, Y_train, do_balance, False)
					X_test, Y_test= balance_samples(X_test, Y_test, do_balance, False)

				cls.n_jobs= mp.cpu_count()
				cls.fit( X_train, Y_train )
				cls.n_jobs= 1
				#score= cls.score( X_test, Y_test )
				Y_pred= cls.predict( X_test )
				score= skmetrics.accuracy_score(Y_test, Y_pred)
				cm += skmetrics.confusion_matrix(Y_test, Y_pred, label_set)
				scores.append( score )
				print('Cross-validation %d / %d (%.2f) - %.1f sec'% (cnum, len(cv), score,timer.sec()) )
				cnum += 1

			# show confusion matrix
			cm_sum= np.sum(cm, axis=1)
			cm_rate= cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
			print('\nY: ground-truth, X: predicted')
			for l in label_set:
				print('%-5s'% cfg.tdef.by_value[l][:5], end='\t')
			print()
			for r in cm_rate:
				for c in r:
					print('%-5.2f'% c, end='\t')
				print()
			print('Average accuracy: %.2f'% np.mean(scores) )
			#X_data= X_data.reshape(1, X_data.shape[0], X_data.shape[1])
			sys.exit()

		else:
			print('>> ERROR: %s not supported yet.'% cfg.FEATURES)
			sys.exit()

		psdparams= cfg.PSD
		for ev in triggers:
			n_epochs[ev]= len( np.where(events[:,-1]==triggers[ev])[0] )

	# Init a classifier
	if cfg.CLASSIFIER=='RF':
		# Make sure to set n_jobs=cpu_count() for training and n_jobs=1 for testing.
		cls= RandomForestClassifier(n_estimators=cfg.RF['trees'], max_features='auto',\
			max_depth=cfg.RF['maxdepth'], n_jobs=mp.cpu_count(), class_weight='balanced' )
	elif cfg.CLASSIFIER=='LDA':
		cls= LDA()
	elif cfg.CLASSIFIER=='rLDA':
		cls= rLDA(cfg.RLDA_REGULARIZE_COEFF)
	else:
		raise RuntimeError, '*** Unknown classifier %s'% cfg.CLASSIFIER

	# Cross-validation
	if cfg.CV_PERFORM is not None:
		ntrials, nsamples, fsize= X_data.shape

		if cfg.CV_PERFORM=='LeaveOneOut':
			print('\n>> %d-fold leave-one-out cross-validation'% ntrials)
			cv= LeaveOneOut(len(Y_data))
		elif cfg.CV_PERFORM=='StratifiedShuffleSplit':
			print('\n>> %d-fold stratified cross-validation with test set ratio %.2f'% (cfg.CV_FOLDS, cfg.CV_TEST_RATIO))
			cv= StratifiedShuffleSplit(Y_data[:,0], cfg.CV_FOLDS, test_size=cfg.CV_TEST_RATIO, random_state=0)
		else:
			print('>> ERROR: Unsupported CV method yet.')
			sys.exit(-1)
		print('%d trials, %d samples per trial, %d feature dimension'% (ntrials, nsamples, fsize) )

		# Do it!
		scores= crossval_epochs(cv, X_data, Y_data, cls, cfg.tdef.by_value, do_balance)

		# Results
		print('\n>> Class information')
		for ev in np.unique(Y_data):
			print('%s: %d trials'% (cfg.tdef.by_value[ev], len(np.where(Y_data[:,0]==ev)[0])) )
		if do_balance:
			print('The number of samples was balanced across classes. Method:', do_balance)

		print('\n>> Experiment conditions')
		print('Spatial filter: %s (channels: %s)'% (spfilter, spchannels) )
		print('Spectral filter: %s'% tpfilter)
		print('Notch filter: %s'% cfg.NOTCH_FILTER)
		print('Channels: %s'% picks)
		print('PSD range: %.1f - %.1f Hz'% (psdparams['fmin'], psdparams['fmax']) )
		print('Window step: %.1f msec'% (1000.0 * psdparams['wstep'] / sfreq) )
		if type(wlen) is list:
			for i, w in enumerate(wlen):
				print('Window size: %.1f sec'% (w) )
				print('Epoch range: %s sec'% (cfg.EPOCH[i]))
		else:
			print('Window size: %.1f sec'% (psdparams['wlen']) )
			print('Epoch range: %s sec'% (cfg.EPOCH))

		#chance= 1.0 / len(np.unique(Y_data))
		cv_mean, cv_std= np.mean(scores), np.std(scores)
		print('\n>> Average CV accuracy over %d epochs'% ntrials)
		if cfg.CV_PERFORM in ['LeaveOneOut','StratifiedShuffleSplit']:
			print("mean %.3f, std: %.3f" % (cv_mean, cv_std) )
		print('Classifier: %s'% cfg.CLASSIFIER)
		if cfg.CLASSIFIER=='RF':
			print('            %d trees, %d max depth'% (cfg.RF['trees'], cfg.RF['maxdepth']) )

		if cfg.USE_LOG:
			logfile= '%s/result_%s_%s.txt'% (datadir, cfg.CLASSIFIER, txt)
			logout= open(logfile, 'a')
			logout.write('%s\t%.3f\t%.3f\n'% (ftrain[0], np.mean(scores), np.var(scores)) )
			logout.close()

	# Train classifier
	archtype= platform.architecture()[0]

	clsfile= '%s/classifier/classifier-%s.pcl'% (datadir,archtype)
	print('\n>> Training classifier')
	X_data_merged= np.concatenate( X_data )
	Y_data_merged= np.concatenate( Y_data )
	if do_balance:
		X_data_merged, Y_data_merged= balance_samples(X_data_merged, Y_data_merged, do_balance, verbose=True)

	timer= qc.Timer()
	cls.fit( X_data_merged, Y_data_merged)
	print('Trained %d samples x %d dimension in %.1f sec'% \
		(X_data_merged.shape[0], X_data_merged.shape[1], timer.sec()))
	# set n_jobs = 1 for testing
	cls.n_jobs= 1

	if cfg.EXPORT_CLS==True:
		classes= { c:cfg.tdef.by_value[c] for c in np.unique(Y_data) }
		if cfg.FEATURES=='PSD':
			data= dict( cls=cls, psde=psde, sfreq=sfreq, picks=picks, classes=classes,
				epochs=cfg.EPOCH, w_frames=w_frames, w_seconds=psdparams['wlen'],
				wstep=psdparams['wstep'], spfilter=spfilter, spchannels=spchannels, refchannel=None,
				tpfilter=tpfilter, notch=cfg.NOTCH_FILTER, triggers=cfg.tdef )
		elif cfg.FEATURES=='TIMELAG':
			data= dict(cls=cls, parameters=cfg.TIMELAG)

		qc.make_dirs('%s/classifier'% datadir)
		qc.save_obj(clsfile, data)

	# Show top distinctive features
	if cfg.CLASSIFIER=='RF' and cfg.FEATURES=='PSD':
		print('\n>> Good features ordered by importance')
		keys, _= qc.sort_by_value( list(cls.feature_importances_), rev=True )
		if cfg.EXPORT_GOOD_FEATURES:
			gfout= open('%s/good_features.txt'% datadir, 'w')

		# reverse-lookup frequency from fft
		if type(wlen) is not list:
			fq= 0
			fq_res= 1.0 / psdparams['wlen']
			fqlist= []
			while fq <= psdparams['fmax']:
				if fq >= psdparams['fmin']: fqlist.append(fq)
				fq += fq_res

			for k in keys[:cfg.FEAT_TOPN]:
				ch,hz= feature2chz(k, fqlist, picks, ch_names=raw.ch_names)
				print('%s, %.1f Hz  (feature %d)'% (ch,hz,k) )
				if cfg.EXPORT_GOOD_FEATURES:
					gfout.write( '%s\t%.1f\n'% (ch, hz) )
			
			if cfg.EXPORT_GOOD_FEATURES:
				if cfg.CV_PERFORM is not None:
					gfout.write('\nCross-validation performance: mean %.2f, std %.2f\n'%(cv_mean, cv_std) )
				gfout.close()
			print()
		else:
			print('Ignoring good features because of multiple epochs.')

	# Test file
	if len(cfg.ftest) > 0:
		raw_test, events_test= pu.load_raw('%s'%(cfg.ftest), spfilter)

		'''
		TODO: implement multi-segment epochs
		'''
		if type(cfg.EPOCH[0]) is list:
			print('MULTI-SEGMENT EPOCH IS NOT SUPPORTED YET.')
			sys.exit(-1)

		epochs_test= Epochs(raw_test, events_test, triggers, tmin=cfg.EPOCH[0], tmax=cfg.EPOCH[1],\
			proj=False, picks=picks, baseline=None, preload=True, add_eeg_ref=False)

		if cfg.FEATURES=='PSD':
			psdfile= 'psd-test.pcl'
			if not os.path.exists(psdfile):
				print('\n>> Computing PSD for test set')
				X_test, y_test= pu.get_psd(epochs_test, psde, w_frames, int(sfreq/8))
				qc.save_obj(psdfile, {'X':X_test, 'y':y_test})
			else:
				print('\n>> Loading %s'% psdfile)
				data= qc.load_obj(psdfile)
				X_test, y_test= data['X'], data['y']
		else:
			print('>> Feature not supported yet for testing set.')
			sys.exit(-1)

		score_test= cls.score( np.concatenate(X_test), np.concatenate(y_test) )
		print('Testing score', score_test)

		# running performance
		print('\nRunning performance over time')
		scores_windows= []
		timer= qc.Timer()
		for ep in range( y_test.shape[0] ):
			scores= []
			frames= X_test[ep].shape[0]
			timer.reset()
			for t in range(frames):
				X= X_test[ep][t,:]
				y= [y_test[ep][t]]
				scores.append( cls.score(X, y) )
				#print('%d /%d   %.1f msec'% (t,X_test[ep].shape[0],1000*timer.sec()) )
			print('Tested epoch %d, %.3f msec per window'%(ep, timer.sec()*1000.0/frames) )
			scores_windows.append(scores)
		scores_windows= np.array(scores_windows)

		###############################################################################
		# Plot performance over time
		###############################################################################
		#w_times= (w_start + w_frames / 2.) / sfreq + epochs.tmin
		step= float(epochs_test.tmax - epochs_test.tmin) / scores_windows.shape[1]
		w_times= np.arange( epochs_test.tmin, epochs_test.tmax, step )
		plt.plot(w_times, np.mean(scores_windows, 0), label='Score')
		plt.axvline(0, linestyle='--', color='k', label='Onset')
		plt.axhline(0.5, linestyle='-', color='k', label='Chance')
		plt.xlabel('time (s)')
		plt.ylabel('Classification accuracy')
		plt.title('Classification score over time')
		plt.legend(loc='lower right')
		plt.show()

	'''
	# CSP-LDA
	elif cfg.CLASSIFIER=='CSPLDA':
		from mne.decoding import CSP
		from mne.layouts import read_layout

		epochs_data= epochs.get_data()
		epochs_train_data= epochs_train.get_data()

		cls= LDA()
		csp= CSP(n_components=16, reg=None, log=True)

		# Use scikit-learn Pipeline with cross_val_score function
		clf= Pipeline([('CSP', csp), ('SVC', cls)])
		scores= cross_val_score(clf, epochs_train_data, Y_data, cv=cv, n_jobs=1)

		# Printing the results
		class_balance= np.mean(Y_data == Y_data[0])
		class_balance= max(class_balance, 1. - class_balance)
		print("Classification accuracy: %f / Chance level: %f" % (np.mean(scores), class_balance))

		# plot CSP patterns estimated on full data for visualization
		csp.fit_transform(epochs_data, Y_data)

		evoked= epochs.average()
		evoked.data= csp.patterns_.T
		evoked.times= np.arange(evoked.data.shape[1])

		layout= read_layout('EEG1005')
		evoked.plot_topomap(times=list(evoked.times), ch_type='eeg', layout=layout,
							scale_time=1, time_format='%i', scale=1,
							unit='Patterns (AU)', size=1.5)

		scores_windows= []

		for train_idx, test_idx in cv:
			Y_data, y_test= Y_data[train_idx], Y_data[test_idx]

			X_data= csp.fit_transform(epochs_train_data[train_idx], Y_data)
			X_test= csp.transform(epochs_train_data[test_idx])

			# fit classifier
			cls.fit(X_data, Y_data)

			# running classifier: test classifier on sliding window
			score_this_window= []
			for n in w_start:
				X_test= csp.transform(epochs_data[test_idx][:, :, n:(n + w_frames)])
				score_this_window.append(cls.score(X_test, y_test))
			scores_windows.append(score_this_window)

		# leeq - train using all
		print('Training using all data')
		X_data= csp.fit_transform(epochs_train_data, Y_data)
		cls.fit(X_data, Y_data)
		model_file= datadir+'/%s-csp-lda.model'%subject
		qc.save_obj(model_file, {'svc':cls, 'csp':csp})
		print('Model saved to %s'% model_file)
	'''

if __name__=='__main__':
	# load parameters
	import imp
	if len(sys.argv) < 2:
		cfg_module= raw_input('Config file name? ')
	else:
		cfg_module= sys.argv[1]
	cfg= imp.load_source(cfg_module, cfg_module)
	#if cfg_module[-3:]=='.py':
	#	cfg_module= cfg_module[:-3]
	#cfg= importlib.import_module(cfg_module)

	# get train list
	ftrain= []
	for f in qc.get_file_list(cfg.DATADIR, fullpath=True):
		if f[-4:] in ['.fif','.fiff','.pcl','.bdf','.gdf']:
			ftrain.append(f)

	# single run
	if True:
		run_trainer(cfg, ftrain, interactive=True)
		sys.exit()

	'''
	# RF parameter search
	if False:
		ftrain= []
		for f in qc.get_file_list(cfg.DATADIR, fullpath=False):
			if f[-4:] != '.gdf':
				print('Skip', f)
				continue
			if not os.path.exists(cfg.DATADIR+f+'.cva'):
				print('Skip', f)
				continue
			print('>> Adding', f)
			ftrain.append(f)

		for rf_trees in [10,50,100,500,1000,2000,5000,10000]:
			for rf_maxdepth in [10,30,50,70,100,200,500,1000,2000]:
				run_trainer(cfg, ftrain)
		sys.exit()
	'''