from __future__ import print_function
from __future__ import division

import socket, time, os, sys
import pycnbi_config
import q_common as qc

class GlassControl():
	"""
	Controls Glass UI

	Constructor:
		mock: set to False if you don't have a Glass.

	"""
	def __init__(self, mock=False):
		self.BUFFER_SIZE = 1024
		self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
		self.last_dir= 'L'
		self.timer= qc.Timer(autoreset=True)
		self.mock= mock
		if self.mock:
			self.print('Using a fake, mock Glass control object.')

	def print(self, *args):
		if len(args) > 0: print('[GlassControl] ', end='')
		print(*args)

	def connect(self, ip, port):
		if self.mock: return
		self.ip= ip
		self.port= port

		# networking via USB if IP=127.0.0.1
		if ip=='127.0.0.1':
			exe= 'adb forward tcp:%d tcp:%d'% (port, port)
			self.print(exe)
			os.system(exe)
			time.sleep(0.2)
		self.print('Connecting to %s:%d'%(ip,port) )
		try:
			self.s.connect((self.ip, self.port))
		except:
			self.print('**** ERROR connecting to Glass. The error was:')
			self.print( sys.exc_info()[0], sys.exc_info()[1] )
			sys.exit(-1)
	
	def disconnect(self):
		if self.mock: return
		self.print('Disconnecting from Glass')
		self.s.close()

	# '\n' is added to the msg
	def send_msg(self, msg, wait=True):
		"""
		Send a message to Glass

		Glass requires some delay after when the last command was sent.
		This function will be blocked until minimum this delay is satisfied.
		Set wait=False to force sending message, but the msg is likely to be ignored.

		"""

		if wait:
			# only wait if the time hasn't passed enough
			self.timer.sleep_atleast(0.033) # 30 Hz

		if self.mock: return

		try:
			#self.s.sendall(bytes(msg, "UTF-8")) # for Python3
			self.s.sendall(bytes( unicode(msg+'\n') ))
		except:
			self.print('**** ERROR: Glass communication failed! Re-initiating the connection.')
			self.disconnect()
			time.sleep(2)
			self.connect(self.ip, self.port)

	# show empty bars
	def clear(self):
		if self.mock: return
		self.send_msg('C')

	# only one direction at a time
	def move_bar(self, new_dir, amount, overlay=False):
		if self.mock: return
		if not overlay and self.last_dir != new_dir:
			self.send_msg('%s0'% self.last_dir)
		self.send_msg( '%s%d'% (new_dir,amount) )
		self.last_dir= new_dir
	
	# fill screen with a solid color (None, 'R','G','B')
	def fill(self, color=None):
		if self.mock: return
		if color==None:
			self.send_msg('F0')
		elif color=='R':
			self.send_msg('F1')
		elif color=='G':
			self.send_msg('F2')
		elif color=='B':
			self.send_msg('F3')
		elif color=='K':
			self.send_msg('F4')

	def fullbar_color(self, color):
		if color not in ['R','G','B','Y']:
			print('**** UNSUPPORTED GLASS BAR COLOR ****')
		else:
			msg= 'B'+color[0]
			#print('*** GLASS SENDING', msg)
			self.send_msg(msg)

if __name__=='__main__':
	step= 5

	ui= GlassControl()
	ui.connect('127.0.0.1', 59900)
	ui.clear()

	#################################
	raw_input('Press Enter')
	print("ui.fullbar_color('Y')")
	ui.fullbar_color('Y')
	ui.move_bar('L', 100)
	
	#################################
	raw_input('Press Enter')
	ui.clear()

	for x in range(10):
		print('L move')
		for x in range(0,101,step):
			ui.move_bar('L', x)
		time.sleep(0.2)

		print('R move')
		for x in range(0,101,step):
			ui.move_bar('R', x)
		time.sleep(0.2)
		
		print('U move')
		for x in range(0,101,step):
			ui.move_bar('U', x)
		time.sleep(0.2)
		
		print('D move')
		for x in range(0,101,step):
			ui.move_bar('D', x)
		time.sleep(0.2)

	ui.clear()
	ui.disconnect()
