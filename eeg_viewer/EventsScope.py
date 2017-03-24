
import numpy as np
import pyqtgraph as pg

class EventsScope():

	def __init__(self, plot_handler, x_ticks, y_range):
		self.events_detected = []
		self.events_curves = []
		self.events_text = []
		self.x_ticks = x_ticks
		self.y_range = y_range
		self.plot_handler = plot_handler

	def add(self, event_position, event_name, event_id):
		if (event_name == "TID"):
			color = (255,0,0)
		elif (event_name == "KEY"):
			color = (0, 255, 0)
		elif (event_name == "LPT"):
			color = (0,0,255)
		else:
			color = (255, 255, 255)

		self.events_detected.append(event_position)
		self.events_detected.append(event_id)					
		self.events_curves.append(self.plot_handler.plot(pen=color, x=np.array([self.x_ticks[-1], self.x_ticks[-1]]), y=np.array(self.y_range)))
		text = pg.TextItem(event_name + "(" + str(self.events_detected[-1]) + ")", anchor=(1.1,0), fill=(0,0,0), color=color)
		text.setPos(self.x_ticks[-1], self.y_range[1])
		self.events_text.append(text)
		self.plot_handler.addItem(self.events_text[-1])


	def set_xticks(self, x_ticks):
		self.x_ticks = x_ticks


	def set_yrange(self, y_range):
		self.y_range = y_range

	def update(self, shift):
		# We have to remove those indexes that reached time = 0s 
		delete_indices_e = []
		delete_indices_c = []
		for x in xrange (0, len(self.events_detected), 2):
			self.events_detected[x] += shift
 			if (self.events_detected[x]<0):
 				delete_indices_e.append(x)
 				delete_indices_e.append(x+1)
 				delete_indices_c.append((x/2))
 				self.events_curves[(x/2)].clear()
 				self.plot_handler.removeItem(self.events_text[(x/2)])

		# self.events_detected = [i for j, i in enumerate(self.events_detected) if j not in delete_indices_e]
		# self.events_curves = [i for j, i in enumerate(self.events_curves) if j not in delete_indices_c]
		# self.events_text = [i for j, i in enumerate(self.events_text) if j not in delete_indices_c]

	def update_plot(self):
		for x in xrange (0, len(self.events_detected), 2):
			self.events_curves[(x/2)].setData(	x=np.array([self.x_ticks[self.events_detected[x]], self.x_ticks[self.events_detected[x]]]), y= self.y_range)
			self.events_text[(x/2)].setPos(self.x_ticks[self.events_detected[x]], self.y_range[0])
