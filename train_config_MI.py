# configuration
DataPath = r'E:\data\zy\Records\fif'
Load_PSD = False
PSD= dict(fmin=4, fmax=30, wlen=1, wstep= 128)
LOAD_EVENTS_FILE= None

from triggerdef_16 import TriggerDef
tdef= TriggerDef()

TRIGGER_DEF= {tdef.LEFT_GO, tdef.RIGHT_GO}
EPOCH = [0,4]
DEBUG_STAND_TRIGGERS= False
CHANNEL_PICKS = None
SP_FILTER= 'car'
SP_CHANNELS= CHANNEL_PICKS
TP_FILTER= None
NOTCH_FILTER= None

FEATURES= 'PSD' # PSD | CSP | TIMELAG
EXPORT_GOOD_FEATURES= True
FEAT_TOPN= 30 # show only the top N features

DWT= dict(freqs=[0.5, 1, 2, 3, 4, 5, 8, 10, 15, 20, 25, 30])
EXPORT_DWT= False
TIMELAG= dict(w_frames=10, wstep=5, downsample=100)
CLASSIFIER = 'RF'
EXPORT_CLS= True
RF= dict(trees=400, maxdepth=100)
RLDA_REGULARIZE_COEFF= 0.3

CV_PERFORM= 'StratifiedShuffleSplit'  # 'StratifiedShuffleSplit' | 'LeaveOneOut' | None
CV_TEST_RATIO= 0.2 # ignored if LeaveOneOut
CV_FOLDS= 10
ftest= ''
USE_LOG= False
USE_CVA= False





if __name__ == '__main__':
	ftrain = []
	import preprocessing
	for f in preprocessing.get_file_list(DataPath, fullPath = True):
		if f[-4:] in ['.fif','.fiff']:
			ftrain.append(f)
	print(ftrain)






