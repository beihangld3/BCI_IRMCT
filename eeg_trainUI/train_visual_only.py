from __future__ import print_function
from __future__ import division

TRIGGER_DEVICE= 'USB2LPT'
#TRIGGER_DEVICE = 'FAKE'

# classes
directions= [0,1] # 0:L, 1:R, 2:U, 3:D
#directions= [2,3]

trials_each= 30 # number of trials for each action

# Google Glass
GLASS_USE= False
GLASS_IP= '127.0.0.1' # USB forwarding

# timings
refresh_rate= 0.05 # in seconds; min=0.01

t_init= 5 # time showing: 'Waiting to start', 15 s
t_gap= 2 # time showing: '1/20 trials'
t_cue= 2 # no bar, only red dot
t_dir_ready= 2 # green bar, no move
t_dir= 4 # blue bar

num_directions= len(directions)
num_trials= len(directions) * trials_each
screen_width= 500
screen_height= 500

import pycnbi_config
import sys, os, math, random, time, datetime
import pyLptControl
from triggerdef_16 import TriggerDef
import cv2
import cv2.cv as cv
import numpy as np
import scipy, scipy.signal
import mne.io, mne.viz
import q_common as qc
import bgi_client

assert refresh_rate >= 0.01

# visualizer
keys= {'left':81,'right':83,'up':82,'down':84,'pgup':85,'pgdn':86,'home':80,'end':87,'space':32,'esc':27\
	,',':44,'.':46,'s':115,'c':99,'[':91,']':93,'1':49,'!':33,'2':50,'@':64,'3':51,'#':35}
img= np.zeros((screen_height,screen_width,3), np.uint8)
color= dict(G=(20,140,0), B=(210,0,0), R=(0,50,200), Y=(0,215,235), K=(0,0,0), W=(255,255,255), w=(200,200,200))

def screen_erase(img):
	cv2.rectangle( img, (0,0), (500,500), (0,0,0), -1 )

# draw cue with custom colors
def draw_cue(img, box=color['R'], cross=color['W']):
	cv2.rectangle( img, (100,200), (400,300), color['w'], -1 )
	cv2.rectangle( img, (200,100), (300,400), color['w'], -1 )
	cv2.rectangle( img, (200,200), (300,300), box, -1 )
	#cv2.circle( img, (250,250), 10, color['Y'], -1 )
	cv2.rectangle( img, (240,248), (260,252), cross, -1 )
	cv2.rectangle( img, (248,240), (252,260), cross, -1 )

# Glass
glass= bgi_client.GlassControl(mock= not GLASS_USE)
glass.connect('127.0.0.1', 59900)
glass.clear()


sps_txt= ''
num_samples= 0
segments= []

dir_sequence= []
for x in range( trials_each ):
	dir_sequence.extend( directions )
random.shuffle( dir_sequence )
event= 'start'
trial= 1

# Hardware trigger
#trigger= pyLptControl.Trigger(TRIGGER_DEVICE)
trigger= pyLptControl.Trigger('USB2LPT', 0x378)
if trigger.init(50)==False:
	print('\n# Error connecting to USB2LPT device. Use a mock trigger instead?')
	raw_input('Press Ctrl+C to stop or Enter to continue.')
	trigger= pyLptControl.MockTrigger()
	trigger.init(50)

cv2.namedWindow("mi")
cv2.moveWindow("mi", 430, 130)
timer_trigger= qc.Timer()
timer_dir= qc.Timer()
timer_refresh= qc.Timer()
tdef= TriggerDef()

# start
while trial <= num_trials:
	timer_refresh.sleep_atleast(0.2)
	#print(timer_refresh.sec())
	timer_refresh.reset()

	# segment= { 'cue':(s,e), 'dir':(s,e), 'label':0-4 } (zero-based)
	if event=='start' and timer_trigger.sec() > t_init:
		event= 'gap_s'
		screen_erase(img)
		timer_trigger.reset()
		trigger.signal(tdef.INIT)
	elif event=='gap_s':
		cv2.putText(img, 'Trial %d / %d'%(trial,num_trials), (150,250), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
		event= 'gap'
	elif event=='gap' and timer_trigger.sec() > t_gap:
		event= 'cue'
		screen_erase(img)
		glass.clear()
		draw_cue(img)
		trigger.signal(tdef.CUE)
		timer_trigger.reset()
	elif event=='cue' and timer_trigger.sec() > t_cue:
		event= 'dir_r'
		dir= dir_sequence[trial-1]
		if dir==0: # left
			glass.move_bar('L', 100)
			cv2.rectangle( img, (100,200), (200,300), color['B'], -1)
			trigger.signal(tdef.LEFT_READY)
		elif dir==1: # right
			glass.move_bar('R', 100)
			cv2.rectangle( img, (300,200), (400,300), color['B'], -1)
			trigger.signal(tdef.RIGHT_READY)
		elif dir==2: # up
			glass.move_bar('U', 100)
			cv2.rectangle( img, (200,100), (300,200), color['B'], -1)
			trigger.signal(tdef.UP_READY)
		elif dir==3: # down
			glass.move_bar('D', 100)
			cv2.rectangle( img, (200,300), (300,400), color['B'], -1)
			trigger.signal(tdef.DOWN_READY)
		timer_trigger.reset()
	elif event=='dir_r' and timer_trigger.sec() > t_dir_ready:
		screen_erase(img)
		draw_cue(img, box=(0,170,0) )
		event= 'dir'
		timer_trigger.reset()
		timer_dir.reset()
		if dir==0: # left
			trigger.signal(tdef.LEFT_GO)
		elif dir==1: # right
			trigger.signal(tdef.RIGHT_GO)
		elif dir==2: # up
			trigger.signal(tdef.UP_GO)
		elif dir==3: # down
			trigger.signal(tdef.DOWN_GO)
	elif event=='dir' and timer_trigger.sec() > t_dir:
		event= 'gap_s'
		screen_erase(img)
		trial += 1
		trigger.signal(tdef.BLANK)
		timer_trigger.reset()

	# protocol
	if event=='dir':
		dx= min( 100, int( 100.0 * timer_dir.sec() / t_dir ) + 1 )
		if dir==0: # L
			glass.move_bar('L', dx)
			cv2.rectangle( img, (200-dx,200), (200,300), color['B'], -1 )
		if dir==1: # R
			glass.move_bar('R', dx)
			cv2.rectangle( img, (300,200), (300+dx,300), color['B'], -1 )
		if dir==2: # U
			glass.move_bar('U', dx)
			cv2.rectangle( img, (200,200-dx), (300,200), color['B'], -1 )
		if dir==3: # D
			glass.move_bar('D', dx)
			cv2.rectangle( img, (200,300), (300,300+dx), color['B'], -1 )

	# wait for start
	if event=='start':
		cv2.putText(img, 'Waiting to start', (120,250), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
	cv2.imshow("mi", img)
	key= 0xFF & cv2.waitKey(1)

	if key==keys['esc']:
		break

cv2.destroyWindow("mi")
if GLASS_USE:
	glass.disconnect()

