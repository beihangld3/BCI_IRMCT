from __future__ import print_function
from __future__ import division
"""
Kyuhwa Lee, CNBI, EPFL, 2016
Dong Liu, IRMCT, BUAA, 2017
"""
# communica with the exoskeleton?
WITH_Robot= True
# classifier
CLS_FILE= r'E:\DongLiu_data\MI\fif\classifier\classifier-64bit.pcl'

# protocol: 
trials_each= 30   # put it to a small number for debug  30
host= '192.168.2.111'
port = 9999

############################################
# For Debug, fake decoder, FAKE_CLS = None or 'left' or 'right' or 'middle' or 'random'
#FAKE_CLS = None  # None
FAKE_CLS = None   # None
# use Google Glass? Always False !!
GLASS_USE= False
# For biosemi:
#AMP_NAME = 'BioSemi'
#AMP_SERIAL  = None
# For debug:
AMP_NAME = None
AMP_SERIAL  = None
PARADIGM= 'BASIC'
CLS_TYPE= 'MI' # 'ERD' or 'ERD_RF' or 'MI'
POSITIVE_FEEDBACK= None # None or 'L' or 'R'

if POSITIVE_FEEDBACK=='L':
	CLS_LEFT= 'LEFT_GO'
	CLS_RIGHT= 'LEFT_READY'
elif POSITIVE_FEEDBACK=='R':
	CLS_LEFT= 'RIGHT_GO'
	CLS_RIGHT= 'RIGHT_READY'
else:
	CLS_LEFT= 'LEFT_GO'
	CLS_RIGHT= 'RIGHT_GO'
# bar speed
BAR_STEP= 10
# debug flags, print pros...
DEBUG_PROBS = False 
# etc
refresh_rate= 0.05 # in seconds; min=0.01

t_init= 1 # initial waiting time before starting  15
t_wait= 2 # delay until next trial. only used if WITH_Robot == False
# end timings relative to the starting of rest
t_trialstart= 2
t_cue= 2 # cue duration for both ERD and MI
t_classify= 5 # MI or nothing

# Kinect4Rex joystitck constants, corresponding to bar_dirs.values()
# no movement, front, back, left, right, stand up, sit down
robot_control= {'L':'{"id"=1,"name"="ld","password"=123,"command"="left"}', 'R': '{"id"=1,"name"="ld","password"=123,"command"="right"}'}

# direction affordances (order: - , + )

directions= ['L', 'R']

# 0-7: from RexController: sum of [1:Walkable, 2:L-turnable, 4:R-turnable]
bar_dirs= {0: [], 1:['U'], 2:['L'], 3:['L','U'], 4:['R'], 5:['U','R'], 6:['L','R'], 7:['L','R','U'], 8:['D','U']}
bar_type= bar_dirs[6] # initial directions for classification


# visualizer
screen_width= 500
screen_height= 500

keys= {'left':81,'right':83,'up':82,'down':84,'pgup':85,'pgdn':86,'home':80,'end':87,'space':32,'esc':27\
	,',':44,'.':46,'s':115,'c':99,'[':91,']':93,'1':49,'!':33,'2':50,'@':64,'3':51,'#':35}


import sys, os, math, random, time, datetime
from datetime import datetime
import pycnbi_config, pyLptControl, bgi_client, cnbi_lsl
import q_common as qc
import numpy as np
import pylsl as lsl
import cv2
import cv2.cv as cv
import mne
from triggerdef_16 import TriggerDef
from stream_receiver import StreamReceiver
from decoder import BCIDecoderDaemon
import socket
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)   #dongliu
s.connect((host, port))
#data = s.recv(1024)
#print('server:'  + data)
s.sendall('{"id"=1,"name"="ld","password"=123,"command"="reset"}')

#dir3 = '{"id"=1,"name"="ld","password"=123,"command"="stop"}'

assert refresh_rate >= 0.01

# define sequence of directions
dir_sequence= []
for x in range( trials_each ):
	dir_sequence.extend( directions )
random.shuffle( dir_sequence )
num_trials= len(directions) * trials_each  # 60


# overlay the bar visual
class Bars:
	"""
	Visualize the bars

	"""
	color= dict(G=(20,140,0), B=(210,0,0), R=(0,50,200), Y=(0,215,235), K=(0,0,0),\
		W=(255,255,255), w=(200,200,200))

	def __init__(self, img, use_glass=False, glass_feedback=True, pc_feedback=True):
		"""
		Params:
		use_glass: if False, mock Glass will be used
		glass_feedback: show feedback to the user?
		pc_feedback: show feedback on the pc screen?

		"""
		import cv2
		self.img= img
		self.glass= bgi_client.GlassControl(mock=not use_glass)
		self.glass.connect('127.0.0.1', 59900)
		self.set_glass_feedback(glass_feedback)
		self.set_pc_feedback(pc_feedback)
		self.set_cue_color(boxcol='R', crosscol='W')

	def finish(self):
		self.glass.disconnect()

	def set_glass_feedback(self, fb):
		self.glass_feedback= fb

	def set_pc_feedback(self, fb):
		self.pc_feedback= fb

	def set_cue_color(self, boxcol='R', crosscol='W'):
		self.boxcol= self.color[boxcol]
		self.crosscol= self.color[crosscol]

	def fill(self, fillcolor='K'):
		self.glass.fill(fillcolor)
		cv2.rectangle( self.img, (0,0), (500,500), self.color[fillcolor], -1 )

	def glass_clear(self):
		self.glass.clear()
	
	# draw cue with custom colors
	def draw_cue(self):
		cv2.rectangle( self.img, (100,200), (400,300), self.color['w'], -1 )
		cv2.rectangle( self.img, (200,100), (300,400), self.color['w'], -1 )
		cv2.rectangle( self.img, (200,200), (300,300), self.boxcol, -1 )
		cv2.rectangle( self.img, (240,248), (260,252), self.crosscol, -1 )
		cv2.rectangle( self.img, (248,240), (252,260), self.crosscol, -1 )

	# paints the new bar on top of the current image
	def move( self, dir, dx, overlay=False ):
		if not overlay:
			self.draw_cue()

		if dx==100: c= self.color['G']
		else: c= self.color['B']

		if dir=='L':
			if self.pc_feedback:
				cv2.rectangle( self.img, (200-dx,200), (200,300), c, -1 )
			if self.glass_feedback:
				self.glass.move_bar( dir, dx, overlay )
		elif dir=='U':
			if self.pc_feedback:
				cv2.rectangle( self.img, (200,200-dx), (300,200), c, -1 )
			if self.glass_feedback:
				self.glass.move_bar( dir, dx, overlay )
		elif dir=='R':
			if self.pc_feedback:
				cv2.rectangle( self.img, (300,200), (300+dx,300), c, -1 )
			if self.glass_feedback:
				self.glass.move_bar( dir, dx, overlay )
		elif dir=='D':
			if self.pc_feedback:
				cv2.rectangle( self.img, (200,300), (300,300+dx), c, -1 )
			if self.glass_feedback:
				self.glass.move_bar( dir, dx, overlay )
		else:
			print('*** ERROR: Unknown direction %s'% dir)

if __name__=='__main__':
	# load trigger definitions
	tdef= TriggerDef()

	# connect hardware trigger
#	trigger= pyLptControl.Trigger('FAKE') # 'USB2LPT'
	trigger= pyLptControl.Trigger('USB2LPT', 0x378)  # dongliu
	if trigger.init(50)==False:
		print('\n# Error connecting to LPT port. Use a mock trigger instead?')
		raw_input('Press Ctrl+C to stop or Enter to continue.')
		trigger= pyLptControl.MockTrigger()
		trigger.init(50)

	subjectID= raw_input('Subject ID ? ').strip()
	print ('subjectID:', subjectID)
	timestamp= datetime.now()
	timestamp= timestamp.strftime('%Y%m%d%H%M%S')

	if not os.path.exists(subjectID):
		os.mkdir(subjectID)
	bar_grand = open(subjectID+'/' + subjectID + '_bar_grand_' + timestamp + '.txt','w')
	for item in dir_sequence:
		bar_grand.write("%s\n" % item)
	bar_grand.close()
	bar_cls = []

	# search LSL server
	amp_name, amp_serial= pu.search_lsl()
	amp_name, amp_serial= None, None
#	amp_name = AMP_NAME
#	amp_serial  = AMP_SERIAL

	# load classifier
	if 'MI' in CLS_TYPE:
		decoder= BCIDecoderDaemon(CLS_FILE, buffer_size=1.0, fake=(FAKE_CLS != None), amp_serial=amp_serial, amp_name=amp_name)
		labels= [tdef.by_value[x] for x in decoder.get_labels()]
		probDown= labels.index(CLS_LEFT)
		probUp= labels.index(CLS_RIGHT)
	else:
		decoder= None

	# init
	event= 'start'
	trial= 1
	num_samples= 0
	sps_txt= ''
	segments= []
	cval= None

	# timers
	timer= qc.Timer()
	timer_refresh= qc.Timer(autoreset=True)

	# visualization
	img= np.zeros((screen_height,screen_width,3), np.uint8)
	bar= Bars(img, GLASS_USE)
	cv2.putText(img, 'Waiting to start', (120,250), cv2.FONT_HERSHEY_SIMPLEX, 1, bar.color['W'], 2)
	cv2.namedWindow("mi")
	cv2.moveWindow("mi", 100, 20)
	cv2.imshow("mi", img)
	bar.fill('G')

	# loop
	tm_cls= qc.Timer()
	p_index= 0
	decision= []
#	while True:
	while (trial<= num_trials):
		timer_refresh.sleep_atleast(0.03)

		""" Start of the protocol """
		if event=='start' and timer.sec() > t_init:
			event= 'wait'
			trigger.signal(tdef.INIT)
			timer.reset()

		""" Rest start """
		'''
		if event=='rest_s':
			timer.reset()
			event= 'rest_go'
			bar.fill('G')
			#cv2.putText(img, 'REST', (150,250), cv2.FONT_HERSHEY_SIMPLEX, 1, bar.color['W'], 2)
			powers_erd= []
		'''

		""" Wait for the next classification timing """
		if event=='wait':
			if timer.sec() > t_wait:
				cval= 6 # U/D      ####  dongliu
				if cval not in bar_dirs:
					print('**** ERROR: unknown bar direction type:', cval)
					sys.exit(-1)

				event= 'classify_init'
				bar_type= list( bar_dirs[cval] ) # always 3-way classification
				timer.reset()

		""" Init classification and show cue """
		if event=='classify_init':
			bar.fill()
			bar.draw_cue()
			bar.glass_clear()
			if timer.sec() > t_trialstart:
				timer.reset()
				event= 'classify_cue'
				dir_now= dir_sequence[trial-1]
				bar.move( dir_now, 100, overlay=False )
				if dir_now=='U':
					trigger.signal( tdef.UP_READY )
				elif dir_now=='D':
					trigger.signal( tdef.DOWN_READY )
				elif dir_now=='L':
					trigger.signal( tdef.LEFT_READY )
				elif dir_now=='R':
					trigger.signal( tdef.RIGHT_READY )
				else:
					raise RuntimeError, 'Unknown direction'
				if CLS_TYPE=='MI':
					probs= [0] * len(labels)
				else:
					erd_acc= 0.0
				dx= 0

		""" Cue finished """
		if event=='classify_cue' and timer.sec() > t_cue:
			timer.reset()
			event= 'classify'
			bar.glass_clear()
			bar.draw_cue()

			if dir_now=='U':
				trigger.signal( tdef.UP_GO )
			elif dir_now=='D':
				trigger.signal( tdef.DOWN_GO )
			elif dir_now=='L':
				trigger.signal( tdef.LEFT_GO )
			elif dir_now=='R':
				trigger.signal( tdef.RIGHT_GO )
			else:
				raise RuntimeError, 'Unknown direction'

		""" Classify """
		if event=='classify':
			if CLS_TYPE=='MI':
				probs_new= decoder.get_prob_unread()
				if probs_new == None:
					if tm_cls.sec() > 3:
						print('>> WARNING: No classification being done. Are you receiving data streams?')
						tm_cls.reset()
					continue
				else:
					tm_cls.reset()

				# accumulate probs
				for i in range( len(probs_new) ):
					probs[i]= probs[i] * 0.9 + probs_new[i] * 0.1 
				probs[probDown]= 0.5
				pdiff= probs[probUp] - probs[probDown]

				if POSITIVE_FEEDBACK=='U':
					if probs[probDown] < probs[probUp]:
						dx += int( pdiff * BAR_STEP )
				elif POSITIVE_FEEDBACK=='D':
					if probs[probDown] > probs[probUp]:
						dx += int( pdiff * BAR_STEP )
				elif POSITIVE_FEEDBACK==None:
					dx += int( pdiff * BAR_STEP )
				if DEBUG_PROBS:
					print(labels, probs_new, pdiff, probs, dx)

			if FAKE_CLS in ['left','down']:
				dx -= 5
			elif FAKE_CLS==['right','up']:
				dx += 5
			elif FAKE_CLS=='middle':
				dx = 15
			dx= max( -100, min(100, dx) )

			if len(bar_type) > 1:
				if dx < 0: # L or U
					bar.move( bar_type[0], abs(dx) )
				else: # U or R
					bar.move( bar_type[1], abs(dx) )

			# classification finished?
			if timer.sec() > t_classify or abs(dx) >= 100: ############>= 
			#if False:
				print('>>>>>>>> Time taken: %.1f'% timer.sec())
				timer.reset()
				event= 'trial_end'
				if dx < 0:
					bar_cls.append('L')
					robot_action= robot_control[ bar_type[0] ]
				else:
					bar_cls.append('R')
					robot_action= robot_control[ bar_type[1] ]

		""" Push decoded action to RexController """
		if event=='trial_end':
			timer.reset()
			event= 'wait'
			print('>> Trial %d finished.'% trial)
			trial += 1			
			bar.fill('G')
			trigger.signal(tdef.BLANK)
			cval= None

			if WITH_Robot:
				print('>> client order', robot_action)
				s.sendall(robot_action)
				time.sleep(5)
			print('>> Waiting for next start signal from the robot.')
			results= []
		
		cv2.imshow("mi", img)
		key= 0xFF & cv2.waitKey(1)
		if key==keys['esc']:
			break

	# clean up
	cv2.destroyWindow("mi")
	bar.finish()

	bar_real = open(subjectID+'/' + subjectID + '_bar_real_' + timestamp + '.txt','w')
	for item in bar_cls:
		bar_real.write("%s\n" % item)
	bar_real.close()
	if decoder: decoder.stop()
	s.sendall('exit')
	s.close()
	print('>> Finished.')

