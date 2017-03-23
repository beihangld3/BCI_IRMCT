# -*- coding: utf-8 -*-
"""
config for training
Kyuhwa Lee, EPFL, 2015
Dong Liu, BUAA, 2017
"""
import pycnbi_config
import numpy as np
DATADIR= r'D:\data\Records\fif'
'''"""""""""""""""""""""""""""
Parameters for computing PSD
Ignored if LOAD_PSD == Ture

wlen: window length in seconds
wstep: window step (32 is enough for 512 Hz, or 256 for 2KHz)

"""""""""""""""""""""""""""'''
LOAD_PSD= False

# ignored if LOAD_PSD==True
#PSD= dict(fmin=1, fmax=30, wlen=2, wstep= 256)
#PSD= dict(fmin=1, fmax=30, wlen=1, wstep= 256)  # wstep,128, 256

PSD= dict(fmin=4, fmax=30, wlen=1, wstep= 128)
#PSD= dict(fmin=8, fmax=13, wlen=1, wstep= 128)
#PSD= dict(fmin=14, fmax=21, wlen=1, wstep= 128)
#PSD= dict(fmin=22, fmax=30, wlen=1, wstep= 128)

'''"""""""""""""""""""""""""""
 EVENTS

 TRIGGER_DEF is ignored if LOAD_PSD==True
"""""""""""""""""""""""""""'''
# None or events filename (hardware events in raw file will be ignored)
# TODO: set this flag in load_multi to concatenate frame numbers in multiple files.
LOAD_EVENTS_FILE= None

from triggerdef_16 import TriggerDef
tdef= TriggerDef()

TRIGGER_DEF= {tdef.LEFT_GO, tdef.RIGHT_GO}
#TRIGGER_DEF= {tdef.UP_GO, tdef.UP_READY}
#TRIGGER_DEF= {tdef.DOWN_GO, tdef.DOWN_READY}
#TRIGGER_DEF= {tdef.UP_READY, tdef.DOWN_READY}

# epoch ranges in seconds relative to onset
#EPOCH= [0, 2] # 这个窗口对magnien来说，取得了75的 sample-based的准确率
EPOCH= [0, 2] # 对于amandine，S3，取得了67%的精度

# change WALk_GO event values
DEBUG_STAND_TRIGGERS= False


'''"""""""""""""""""""""""""""
 CHANNELS

 Pick a subset of channels for PSD. Note that Python uses zero-based indexing.
 However, for fif files saved using PyCNBI library, index 0 is the trigger channel
 and data channels start from index 1. (to be consistent with MATLAB)

 Ignored if LOAD_PSD= True

"""""""""""""""""""""""""""'''
CHANNEL_PICKS= None # use all channels
#CHANNEL_PICKS = [4,5,6,8,9,10,12,13,19,21,22,23,25,26,27,31,32]

'''"""""""""""""""""""""""""""
 FILTERS
"""""""""""""""""""""""""""'''
# apply spatial filter immediately after loading data
# SP_FILTER= None | 'car' | 'laplacian'
SP_FILTER= 'car'
# only consider the following channels while computing
SP_CHANNELS= CHANNEL_PICKS

# apply spectrial filter immediately after applying SP_FILTER
# Can be either overlap-add FIR or forward-backward IIR via filtfilt
# Value: None or [lfreq, hfreq]
# if lfreq < hfreq: bandpass
# if lfreq > hfreq: bandstop
# if lfreq == None: highpass
# if hfreq == None: lowpass
#TP_FILTER= [0.6, 4.0]


TP_FILTER= None
#TP_FILTER= [1, 50]

NOTCH_FILTER= None # None or list of values

'''"""""""""""""""""""""""""""
 FEATURE TYPE
"""""""""""""""""""""""""""'''
FEATURES= 'PSD' # PSD | CSP | TIMELAG
EXPORT_GOOD_FEATURES= True
FEAT_TOPN= 30 # show only the top N features

# Wavelet parameters
#DWT= dict(freqs=[0.5, 1, 2, 3, 4, 5, 8, 10, 18])
DWT= dict(freqs=[0.5, 1, 2, 3, 4, 5, 8, 10, 15, 20, 25, 30])
# export wavelets into MATLAB file
EXPORT_DWT= False


'''"""""""""""""""""""""""""""
 TimeLag parameters

 w_frames: window length in frames (samples) of downsampled data
 wstep: window step in downsampled data
 downsample: average every N-sample block (reduced to 1/N samples)
"""""""""""""""""""""""""""'''
TIMELAG= dict(w_frames=10, wstep=5, downsample=100)


'''"""""""""""""""""""""""""""
 CLASSIFIER
"""""""""""""""""""""""""""'''
# clasifier
#CLASSIFIER= 'RF' # RF | LDA | rLDA
CLASSIFIER = 'RF'
#CLASSIFIER= 'LDA'

EXPORT_CLS= True

# RF parameters
RF= dict(trees=400, maxdepth=100)

# rLDA parameter
RLDA_REGULARIZE_COEFF= 0.3


'''"""""""""""""""""""""""""""
 CROSS-VALIDATION & TESTING
"""""""""""""""""""""""""""'''
# do cross-validation?
CV_PERFORM= 'StratifiedShuffleSplit'  # 'StratifiedShuffleSplit' | 'LeaveOneOut' | None
CV_TEST_RATIO= 0.2 # ignored if LeaveOneOut
CV_FOLDS= 10

# testing file
ftest= ''
#ftest = r'E:\LP_magnien\fif\20160511-093524-raw.fif'

'''"""""""""""""""""""""""""""
 ETC
"""""""""""""""""""""""""""'''
# write to log file?
USE_LOG= False

# use CVA feature selection?
USE_CVA= False
