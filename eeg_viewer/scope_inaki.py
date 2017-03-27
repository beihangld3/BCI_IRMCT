# EEG Scope
# Inaki Iturrate, 07.2015
#
# V0.1 
# TODO
#	- Should move to VisPY: http://vispy.org/plot.html#module-vispy.plot but still under development
#	- The scope should be a class itself
#	- Scope of EXG signals
#	- Events should stored in a class.
#
#

import ConfigParser
import sys, time, math, string
import BCI
from time import *
import pylpttrigger
import os
import struct
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, uic
import numpy as np
import BCI
from scipy.signal import butter, lfilter,lfiltic, buttord
import subprocess
import ConfigParser

path = "/usr/share/eeg_scope/"

# Load GUI. Designed with QT Creator, feel free to change stuff
form_class = uic.loadUiType(path + "mainwindow.ui")[0] 

class Scope(QtGui.QMainWindow, form_class):
    

	def __init__(self):
		super(Scope, self).__init__()
		self.init_scope()


	#
	# 	Main init function
	#
	def init_scope(self):      

		# pg.setConfigOption('background', 'w')
		# pg.setConfigOption('foreground', 'k')

		self.scope_settings = ConfigParser.RawConfigParser(allow_no_value=True)
		if (not os.path.isfile(os.getenv("HOME") + "/.scope_settings.ini")):
			subprocess.Popen(["cp", path + "/.scope_settings.ini", os.getenv("HOME") + "/.scope_settings.ini"], close_fds=True)	
	
		self.scope_settings.read(os.getenv("HOME") + "/.scope_settings.ini")
		self.init_loop()
		self.init_panel_GUI()		
		self.init_scope_GUI()
		self.init_timer()

	#
	# 	Initialize control panel parameter
	#
	def init_panel_GUI(self):
		self.setupUi(self)


		self.show_TID_events = False
		self.show_LPT_events = False
		self.show_Key_events = False

		# Event handler
		self.comboBox_scale.activated.connect(self.onActivated_combobox_scale) 
		self.spinBox_time.valueChanged.connect(self.onValueChanged_spinbox_time) 
		self.checkBox_car.stateChanged.connect(self.onActivated_checkbox_car)
		self.checkBox_bandpass.stateChanged.connect(self.onActivated_checkbox_bandpass)
		self.checkBox_showTID.stateChanged.connect(self.onActivated_checkbox_TID)
		self.checkBox_showLPT.stateChanged.connect(self.onActivated_checkbox_LPT)
		self.checkBox_showKey.stateChanged.connect(self.onActivated_checkbox_Key)
		self.pushButton_bp.clicked.connect(self.onClicked_button_bp)
		self.pushButton_rec.clicked.connect(self.onClicked_button_rec)
		self.pushButton_stoprec.clicked.connect(self.onClicked_button_stoprec)

		self.pushButton_stoprec.setEnabled(False)
		self.comboBox_scale.setCurrentIndex(4)
		self.checkBox_bandpass.setChecked(True)
		self.checkBox_showTID.setChecked(int(self.scope_settings.get("plot", "show_TID_events")))
		self.checkBox_showLPT.setChecked(int(self.scope_settings.get("plot", "show_LPT_events")))
		self.checkBox_showKey.setChecked(int(self.scope_settings.get("plot", "show_KEY_events")))
		self.statusBar.showMessage("[Not recording]")

		self.screen_width = 522
		self.screen_height = 160
		self.setGeometry(100,100, self.screen_width, self.screen_height)
		self.setFixedSize(self.screen_width, self.screen_height)
		self.setWindowTitle('EEG Scope Panel')
		self.setFocusPolicy(QtCore.Qt.ClickFocus)
		self.setFocus()
		self.show()


	#
	#	Initialize scope parameters
	#
	def init_scope_GUI(self):

		self.bool_parser ={True: '1', False: '0'}

		# PyQTGraph plot initialization
		self.win = pg.GraphicsWindow()
		self.win.setWindowTitle('EEG Scope')
		self.win.setWindowFlags(QtCore.Qt.WindowMinimizeButtonHint)
		self.win.keyPressEvent = self.keyPressEvent
		self.win.show()
		self.plot_handler = self.win.addPlot()

		# Scales available in the GUI. If you change the options in the GUI
		# you should change them here as well
		self.scales_range = [1, 10, 25, 50, 100, 250, 500, 1000, 2500]

		# Scale in uV
		self.scale = int(self.scope_settings.get("plot", "scale_plot"))
		# Time window to show in seconds
		self.seconds_to_show = int(self.scope_settings.get("plot", "time_plot"))

		# Y Tick labels. Use values from the config file.
		self.show_channel_names = int(self.scope_settings.get("internal", "show_channel_names"))
		self.channel_labels = []
		values = []
		for x in range(0, self.config['eeg_channels']):
			if (self.show_channel_names):
				self.channel_labels.append(self.scope_settings.get("internal", "channel_names_gtec").split(', ')[x])
				values.append((-x*self.scale, self.channel_labels[-1]))
			else:
				self.channel_labels.append('CH ' + str(x+1))
				values.append((-x*self.scale, self.channel_labels[-1]))
		values_axis = []
		values_axis.append(values)
		values_axis.append([])			

		# Plot initialization
		self.plot_handler.getAxis('left').setTicks(values_axis)		
		self.plot_handler.setRange(xRange=[0, self.seconds_to_show], yRange=[+1.5*self.scale, -0.5*self.scale-self.scale*self.config['eeg_channels']])
		self.plot_handler.disableAutoRange()
		self.plot_handler.showGrid(y=True)
		self.plot_handler.setLabel(axis='left', text='Scale (uV): ' + str(self.scale))
		self.plot_handler.setLabel(axis='bottom', text='Time (s)')

		# X axis
		self.x_ticks = np.zeros(self.config['sf']*self.seconds_to_show);
		for x in range(0, self.config['sf']*self.seconds_to_show):
			self.x_ticks[x] = (x*1)/float(self.config['sf'])

		# EEG data for plotting
		self.data_plot = np.zeros((self.config['sf']*self.seconds_to_show, self.config['eeg_channels']))
		self.curve_eeg = []
		for x in range (0, self.config['eeg_channels']):
			self.curve_eeg.append(self.plot_handler.plot(self.data_plot[:,x]))

		# Plotting colors. If channels > 16, colors will roll back to the beginning
		self.colors = np.array([[255,0,0], [0,255,0], [0,0, 255], [255,255,0], [0,255,255], [255,0,255], 
								[128,100,100], [0,128,0], [0,128, 128], [128,128,0], [255,128,128], [128,0,128], 
								[128,255,0], [255,128,0], [0,255, 128], [128,0, 255]])

		# Events data
		self.events_detected = []
		self.events_curves = []
		self.events_text = []

		# CAR initialization
		self.apply_car = int(self.scope_settings.get("filtering", "apply_car_filter"))
		self.matrix_car = np.zeros((self.config['eeg_channels'], self.config['eeg_channels']), dtype=float)
		self.matrix_car[:,:] = -1/float(self.config['eeg_channels'])
		np.fill_diagonal(self.matrix_car, 1-(1/float(self.config['eeg_channels'])))

		# Laplacian initalization. TO BE DONE
		self.matrix_lap = np.zeros((self.config['eeg_channels'], self.config['eeg_channels']), dtype=float)
		np.fill_diagonal(self.matrix_lap, 1)
		self.matrix_lap[2,0] = -1
		self.matrix_lap[0,2] = -0.25
		self.matrix_lap[0,2] = -0.25

		# BP initialization
		self.apply_bandpass = int(self.scope_settings.get("filtering", "apply_bandpass_filter"))
		if (self.apply_bandpass):
			self.doubleSpinBox_hp.setValue(float(self.scope_settings.get("filtering", "bandpass_cutoff_frequency").split(' ')[0]))
			self.doubleSpinBox_lp.setValue(float(self.scope_settings.get("filtering", "bandpass_cutoff_frequency").split(' ')[1]))
			self.pushButton_bp.click()

		self.checkBox_bandpass.setChecked(self.apply_car)
		self.checkBox_bandpass.setChecked(self.apply_bandpass)

		self.update_title_scope()

		# Help variables
		self.show_help = 0
		self.help = pg.TextItem(
		"CNBI EEG Scope v0.3 \n" + 
		"----------------------------------------------------------------------------------\n" + 
		"C: De/activate CAR Filter\n" + 
		"B: De/activate Bandpass Filter (with current settings)\n" + 
		"T: Show/hide TiD events\n" + 
		"L: Show/hide LPT events\n" + 
		"K: Show/hide Key events. If not shown, they are NOT recorded!\n" + 
		"0-9: Add a user-specific Key event. Do not forget to write down why you marked it.\n" + 
		"Up, down arrow keys: Increase/decrease the scale, steps of 10 uV\n" + 
		"Left, right arrow keys: Increase/decrease the time to show, steps of 1 s\n" + 
		"Spacebar: Stop the scope plotting, whereas data acquisition keeps running (EXPERIMENTAL)\n" + 
		"Esc: Exits the scope",
		anchor=(0,0), border=(70,70,70), fill=pg.mkColor(20,20,20,200), color=(255,255,255))

		# Stop plot functionality
		self.stop_plot = 0

		# Force repaint even when we shouldn't repaint.
		self.force_repaint = 0

		# For some strange reason when the width is > 1 px the scope runs slow.
		# self.pen_plot = []
		# for x in range(0, self.config['eeg_channels']):
		# 	self.pen_plot.append(pg.mkPen(self.colors[x%16,:], width=3))


	#
	# 	Initializes the BCI loop parameters
	#
	def init_loop(self):

		self.fin = open(self.scope_settings.get("internal", "path_pipe"), 'r')		

		# 12 unsigned ints (4 bytes) 
		data = struct.unpack("<12I", self.fin.read(4*12)) 

		self.config = {	'id':data[0], 'sf':data[1], 'labels':data[2], 'samples':data[3], 'eeg_channels':data[4], 'exg_channels':data[5], 
					'tri_channels':data[6], 'eeg_type':data[8], 'exg_type':data[9], 'tri_type':data[10], 'lbl_type':data[11], 
					'tim_size':1, 'idx_size':1}

		self.tri = np.zeros(self.config['samples'])
		self.eeg = np.zeros((self.config['samples'],self.config['eeg_channels']), dtype=np.float)
		self.exg = np.zeros((self.config['samples'],self.config['exg_channels']), dtype=np.float)

		# TID initialization
		self.bci = BCI.BciInterface()
		self.bci.id_msg_bus.SetEvent(400)
		self.bci.iDsock_bus.sendall(self.bci.id_serializer_bus.Serialize())	

	#
	# 	Initializes the QT timer, which will call the update function every 20 ms
	#
	def init_timer(self):

		QtCore.QCoreApplication.processEvents()
		QtCore.QCoreApplication.flush()
		self.timer = QtCore.QTimer(self)
		self.timer.timeout.connect(self.update_loop)
		self.timer.start(20);


	#
	#	Main update function (connected to the timer)
	#
	def update_loop(self):

		self.handle_tobiid_input()	# Read TiDs
		self.read_eeg() 			# Read new chunk
		self.filter_signal()		# Filter acquired data
		self.update_ringbuffers()	# Update the plotting infor
		if (not self.stop_plot):	
			self.repaint() 			# Call paint event


	#
	#	Read EEG
	#
	def read_eeg(self):

		# Reading in python is blocking, so it will wait until having the amount of data needed
		# Read timestamp. 1 value, type double
		timestamp = struct.unpack("<d", self.fin.read(8*1))
		# Read index. 1 value, type uint64
		index = struct.unpack("<Q", self.fin.read(8*1))
		# Read labels. self.config.labels, type double
		labels = struct.unpack("<" + str(self.config['labels']) + "I", self.fin.read(4*self.config['labels']))
		# Read eeg. self.config.samples*self.config.eeg_ch, type float
		beeg = struct.unpack("<" + str(self.config['samples']*self.config['eeg_channels']) + "f", self.fin.read(4*self.config['samples']*self.config['eeg_channels']))
		self.eeg = np.reshape(list(beeg), (self.config['samples'],self.config['eeg_channels']))
		# Read exg. self.config.samples*self.config.exg_ch, type float
		bexg = struct.unpack("<" + str(self.config['samples']*self.config['exg_channels']) + "f", self.fin.read(4*self.config['samples']*self.config['exg_channels']))
		self.exg = np.reshape(list(bexg), (self.config['samples'],self.config['exg_channels']))
		# Read tri. self.config.samples*self.config.tri_ch, type float
		self.tri = struct.unpack("<" + str(self.config['samples']*self.config['tri_channels']) + "i", self.fin.read(4*self.config['samples']*self.config['tri_channels']))


	#
	#	Bandpas + CAR filtering
	#
	def filter_signal(self):

		if (self.apply_car):
			self.eeg = np.dot(self.matrix_car, np.transpose(self.eeg))
			self.eeg = np.transpose(self.eeg)

		if (self.apply_bandpass):
			for x in range (0, self.eeg.shape[1]):
				self.eeg[:,x], self.zi[:,x] = lfilter(self.b, self.a, self.eeg[:,x], -1, self.zi[:,x])


	#
	#	Update ringbuffers and events for plotting
	#
	def update_ringbuffers(self):


		# EEG ringbuffer rolling
		self.data_plot=np.roll(self.data_plot, -self.config['samples'], 0)
		self.data_plot[-self.config['samples']:, :] = self.eeg

		# We have to remove those indexes that reached time = 0  
		delete_indices_e = []
		delete_indices_c = []
		for x in xrange (0, len(self.events_detected), 2):
			self.events_detected[x] -= self.config['samples'] # shift event
 			if (self.events_detected[x]<0) & (not self.stop_plot):
 				delete_indices_e.append(x)
 				delete_indices_e.append(x+1)
 				delete_indices_c.append((x/2))
 				self.events_curves[(x/2)].clear()
 				self.plot_handler.removeItem(self.events_text[(x/2)])

		self.events_detected = [i for j, i in enumerate(self.events_detected) if j not in delete_indices_e]
		self.events_curves = [i for j, i in enumerate(self.events_curves) if j not in delete_indices_c]
		self.events_text = [i for j, i in enumerate(self.events_text) if j not in delete_indices_c]


		# Find LPT events and add them
		if (self.show_LPT_events) & (not self.stop_plot):
			for x in range (1, len(self.tri)):
				if (self.tri[x] != self.tri[x-1]):
					if (self.tri[x] != 0):
						self.addEventPlot("LPT", self.tri[x])


	#	
	#	Called by repaint()
	#
	def paintEvent(self, e):
		# Distinguish between paint events from timer and event QT widget resizing, clicking etc (sender == none)
		# Only paint when the timer called.
		sender = self.sender()
		if (sender == None) & (not self.force_repaint):
			pass
		else:
			self.force_repaint = 0
			qp = QtGui.QPainter()
			qp.begin(self)
			# Update the interface
			self.paintInterface(qp)
			qp.end()

       
    #
    #	Update stuff on the interface. Only graphical updates should be added here
    #
	def paintInterface(self, qp):

		# Update EEG channels
		for x in range (0, self.config['eeg_channels']):
			self.curve_eeg[x].setData(x=self.x_ticks, y=self.data_plot[:,x]-x*self.scale,pen=self.colors[x%16,:])
			# self.curve_eeg[x].setData(x=self.x_ticks, y=self.data_plot[:,x]-x*self.scale,pen=self.pen_plot[x])
		
		# Update events
		for x in xrange (0, len(self.events_detected), 2):
			self.events_curves[(x/2)].setData(x=np.array([self.x_ticks[self.events_detected[x]], self.x_ticks[self.events_detected[x]]]), y=np.array([+1.5*self.scale, -0.5*self.scale-self.scale*self.config['eeg_channels']]))
			self.events_text[(x/2)].setPos(self.x_ticks[self.events_detected[x]], +1.5*self.scale)

		
	# 
	#	Do necessary stuff when scale has changed
	#
	def update_plot_scale(self, new_scale):
		
		if (new_scale <= 0):
			new_scale = 1
		else:
			new_scale = new_scale - new_scale%10

		self.scale = new_scale

		# Y Tick labels
		values = []
		for x in range(0, self.config['eeg_channels']):
			values.append((-x*self.scale, self.channel_labels[x]))
		values_axis = []
		values_axis.append(values)
		values_axis.append([])			

		self.plot_handler.getAxis('left').setTicks(values_axis)
		self.plot_handler.setRange(yRange=[+self.scale, -self.scale*self.config['eeg_channels']])
		self.plot_handler.setLabel(axis='left', text='Scale (uV): ' + str(self.scale))
		self.trigger_help()

		# Immediate update to avoid "shakiness"
		if (not self.stop_plot):
			self.force_repaint = 1
			self.repaint()

	# 
	#	Do necessary stuff when seconds to show have changed
	#
	def update_plot_seconds(self, new_seconds):

		if (new_seconds != self.seconds_to_show) & (new_seconds > 0) & (new_seconds < 100):
			self.spinBox_time.setValue(new_seconds)
			self.plot_handler.setRange(xRange=[0, new_seconds])
			self.x_ticks = np.zeros(self.config['sf']*new_seconds);
			for x in range(0, self.config['sf']*new_seconds):
				self.x_ticks[x] = (x*1)/float(self.config['sf'])

			if (new_seconds > self.seconds_to_show):
				padded_signal = np.zeros((self.config['sf']*new_seconds, self.config['eeg_channels']))
				padded_signal[padded_signal.shape[0]-self.data_plot.shape[0]:, :] = self.data_plot
				for x in xrange (0, len(self.events_detected), 2):
					self.events_detected[x] += padded_signal.shape[0]-self.data_plot.shape[0]
				self.data_plot = padded_signal

			else:
				for x in xrange (0, len(self.events_detected), 2):
					self.events_detected[x] -= self.data_plot.shape[0]-self.config['sf']*new_seconds
				self.data_plot = self.data_plot[self.data_plot.shape[0]-self.config['sf']*new_seconds:,:]

			self.seconds_to_show = new_seconds
			self.trigger_help()

			# Immediate update to avoid "shakiness"
			if (not self.stop_plot):
				self.force_repaint = 1
				self.repaint()
	#
	# Handle TOBI iD events
	#
	def handle_tobiid_input(self):
		data = None
		try:
			data = self.bci.iDsock_bus.recv(512)
			self.bci.idStreamer_bus.Append(data)	
		except:
			self.nS = False
			self.dec = 0
			pass	

		# deserialize ID message
		if data:
			if self.bci.idStreamer_bus.Has("<tobiid","/>"):
				msg = self.bci.idStreamer_bus.Extract("<tobiid","/>")
				self.bci.id_serializer_bus.Deserialize(msg)
				self.bci.idStreamer_bus.Clear()
				tmpmsg = int(self.bci.id_msg_bus.GetEvent())
				if (self.show_TID_events) & (not self.stop_plot):
					self.addEventPlot("TID", tmpmsg)					
				
			elif self.bci.idStreamer_bus.Has("<tcstatus","/>"):
				MsgNum = self.bci.idStreamer_bus.Count("<tcstatus")
				for i in range(1,MsgNum-1):
					# Extract most of these messages and trash them		
					msg_useless = self.bci.idStreamer_bus.Extract("<tcstatus","/>")       

	#
	# 	Add an event to the scope
	#
	def addEventPlot(self, event_name, event_id):
		if (event_name == "TID"):
			color = (255,0,0)
		elif (event_name == "KEY"):
			color = (0, 255, 0)
		elif (event_name == "LPT"):
			color = (0,0,255)
		else:
			color = (255, 255, 255)

		self.events_detected.append(self.data_plot.shape[0]-1)
		self.events_detected.append(event_id)					
		self.events_curves.append(self.plot_handler.plot(pen=color, x=np.array([self.x_ticks[-1], self.x_ticks[-1]]), y=np.array([+1.5*self.scale, -1.5*self.scale*self.config['eeg_channels']])))
		text = pg.TextItem(event_name + "(" + str(self.events_detected[-1]) + ")", anchor=(1.1,0), fill=(0,0,0), color=color)
		text.setPos(self.x_ticks[-1], +1.5*self.scale)
		self.events_text.append(text)
		self.plot_handler.addItem(self.events_text[-1])


	#
	#	Calculation of bandpass coefficients.
	#	Order is computed automatically.
	#	Note that if filter is unstable this function crashes.
	#
	def butter_bandpass(self, highcut, lowcut, fs, num_ch):
		low = lowcut/(0.5*fs)
		high = highcut/(0.5*fs)
		# get the order. TO BE DONE: Sometimes it fails
		ord = buttord(high, low, 2, 40)
		b, a = butter(ord[0], [high, low], btype='band')
		zi = np.zeros([a.shape[0]-1, num_ch])
		return b, a, zi
		

	#
	#	Updates the title shown in the scope
	#
	def update_title_scope(self):
		if (hasattr(self, 'plot_handler')):
			self.plot_handler.setTitle(title=
				'TLK: ' + self.bool_parser[self.show_TID_events] + self.bool_parser[self.show_LPT_events] + self.bool_parser[self.show_Key_events] + 
				', CAR: ' + self.bool_parser[self.apply_car] + 
				', BP: ' + self.bool_parser[self.apply_bandpass] + ' [' + str(self.doubleSpinBox_hp.value()) + '-' + str(self.doubleSpinBox_lp.value()) + '] Hz')
				# ', BP: ' + self.bool_parser[self.apply_bandpass] + (' [' + str(self.doubleSpinBox_hp.value()) + '-' + str(self.doubleSpinBox_lp.value()) + '] Hz' if self.apply_bandpass else ''))

	#
	#	Shows / hide help in the scope window
	#
	def trigger_help(self):
		if self.show_help:
			self.help.setPos(0, (-self.scale))
			self.plot_handler.addItem(self.help)
			self.help.setZValue(1)
		else:
			self.plot_handler.removeItem(self.help)


	# ----------------------------------------------------------------------------------------------------
	# 			EVENT HANDLERS
	# ----------------------------------------------------------------------------------------------------
	def onClicked_button_rec(self):
		# Simply call cl_rpc for this.
		if (len(self.lineEdit_recFilename.text())>0):
			if ".gdf" in self.lineEdit_recFilename.text():
				self.pushButton_stoprec.setEnabled(True)
				self.pushButton_rec.setEnabled(False)
				# Popen is more efficient than os.open, since it is non-blocking
				subprocess.Popen(["cl_rpc", "openxdf",  str(self.lineEdit_recFilename.text()), "dummy_log", "dummy_log"], close_fds=True)
				self.statusBar.showMessage("Recording file " + str(self.lineEdit_recFilename.text()))
			elif ".bdf" in self.lineEdit_recFilename.text(): 
				self.pushButton_stoprec.setEnabled(True)
				self.pushButton_rec.setEnabled(False)
				subprocess.Popen(["cl_rpc", "openxdf",  str(self.lineEdit_recFilename.text()), "dummy_log", "dummy_log"], close_fds=True)
				self.statusBar.showMessage("Recording file " + str(self.lineEdit_recFilename.text()))
			else:
				pass

	def onClicked_button_stoprec(self):
		subprocess.Popen(["cl_rpc", "closexdf"], close_fds=True)
		self.pushButton_rec.setEnabled(True)
		self.pushButton_stoprec.setEnabled(False)
		self.statusBar.showMessage("Not recording")

	def onActivated_checkbox_bandpass(self):
		self.apply_bandpass = False
		self.pushButton_bp.setEnabled(self.checkBox_bandpass.isChecked())
		self.doubleSpinBox_hp.setEnabled(self.checkBox_bandpass.isChecked())
		self.doubleSpinBox_lp.setEnabled(self.checkBox_bandpass.isChecked())
		self.update_title_scope()

	def onActivated_checkbox_car(self):
		self.apply_car = self.checkBox_car.isChecked()
		self.update_title_scope()

	def onActivated_checkbox_TID(self):
		self.show_TID_events = self.checkBox_showTID.isChecked()
		self.update_title_scope()

	def onActivated_checkbox_LPT(self):
		self.show_LPT_events = self.checkBox_showLPT.isChecked()
		self.update_title_scope()

	def onActivated_checkbox_Key(self):
		self.show_Key_events = self.checkBox_showKey.isChecked()
		self.update_title_scope()

	def onValueChanged_spinbox_time(self):
		self.update_plot_seconds(self.spinBox_time.value())
				
	def onActivated_combobox_scale(self):
		self.update_plot_scale(self.scales_range[self.comboBox_scale.currentIndex()])

	def onClicked_button_bp(self):
		if (self.doubleSpinBox_lp.value() > self.doubleSpinBox_hp.value()):
			self.apply_bandpass = True
			self.b, self.a, self.zi = self.butter_bandpass(self.doubleSpinBox_hp.value(), self.doubleSpinBox_lp.value(), self.config['sf'], self.config['eeg_channels'])
		self.update_title_scope()

	def keyPressEvent(self, event):
		key = event.key()
		if (key == QtCore.Qt.Key_Escape):	
			self.closeEvent(None)
		if (key == QtCore.Qt.Key_H):
			self.show_help = not self.show_help
			self.trigger_help()
		if (key == QtCore.Qt.Key_Up):	
			self.update_plot_scale(self.scale+10)
		if (key == QtCore.Qt.Key_Space):	
			self.stop_plot = not self.stop_plot
		if (key == QtCore.Qt.Key_Down):	
			self.update_plot_scale(self.scale-10)
		if (key == QtCore.Qt.Key_Left):	
			self.update_plot_seconds(self.seconds_to_show-1)
		if (key == QtCore.Qt.Key_Right):	
			self.update_plot_seconds(self.seconds_to_show+1)
		if (key == QtCore.Qt.Key_L):
			self.checkBox_showLPT.setChecked(not self.checkBox_showLPT.isChecked())	
		if (key == QtCore.Qt.Key_T):	
			self.checkBox_showTID.setChecked(not self.checkBox_showTID.isChecked())	
		if (key == QtCore.Qt.Key_K):	
			self.checkBox_showKey.setChecked(not self.checkBox_showKey.isChecked())	
		if (key == QtCore.Qt.Key_C):	
			self.checkBox_car.setChecked(not self.checkBox_car.isChecked())	
		if (key == QtCore.Qt.Key_B):	
			self.checkBox_bandpass.setChecked(not self.checkBox_bandpass.isChecked())	
			if self.checkBox_bandpass.isChecked():
				self.pushButton_bp.click()
		if ((key>=QtCore.Qt.Key_0) & (key<=QtCore.Qt.Key_9)):
			if (self.show_Key_events) & (not self.stop_plot):
				self.addEventPlot("KEY", 990 + key - QtCore.Qt.Key_0)
				self.bci.id_msg_bus.SetEvent(990 + key - QtCore.Qt.Key_0)
				self.bci.iDsock_bus.sendall(self.bci.id_serializer_bus.Serialize());				

	#
	#	Function called when a closing event was triggered.
	#
	def closeEvent(self, event):
		reply = QtGui.QMessageBox.question(self, "Quit", "Are you sure you want to quit?", QtGui.QMessageBox.Yes | QtGui.QMessageBox.No, QtGui.QMessageBox.Yes)
		if (reply == QtGui.QMessageBox.Yes):
			if (self.pushButton_stoprec.isEnabled()):
				subprocess.Popen(["cl_rpc", "closexdf"], close_fds=True)
			self.fin.close()
			exit()

	# ----------------------------------------------------------------------------------------------------	
	# 		END OF EVENT HANDLERS
	# ----------------------------------------------------------------------------------------------------
		


def main():
    
    app = QtGui.QApplication(sys.argv)
    ex = Scope()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
