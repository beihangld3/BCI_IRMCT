from __future__ import print_function
from __future__ import division

'''
Trigger definition class

'by_key' and 'values' member variables are automatically created when instantiated.

Usage: See the sample code


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

'''

# Trigger values up to 15
class TriggerDef:
	"""
	Trigger definition class

	'by_key' and 'values' member variables are automatically created when instantiated.

	Usage: See the sample code
	"""
	INIT= 15 # start of the trial
	CUE= 14 # cue shown
	BLANK= 13 # screen turned into blank
	LEFT_READY= 12 # left bar shown
	LEFT_GO= 11 # left bar started moving
	RIGHT_READY= 10 # right bar shown
	RIGHT_GO= 9 # right started moving
	UP_READY= 8 # up bar shown
	UP_GO= 7 # up started moving
	DOWN_READY= 6 # down bar shown
	DOWN_GO= 5 # down started moving
	BOTH_READY= 4 # left+right
	BOTH_GO= 3 # left+right
	FEEDBACK= 2 # feedback shown
	GO= 1 # generic go signal

	def __init__(self):
		for attr in dir(self):
			if hasattr(self, 'by_value')==False:
				self.by_key= {}
				self.by_value= {}
			if not callable(getattr(self,attr)) and not attr.startswith("__"):
				#print(attr, getattr(TriggerDef,attr))
				self.by_key[attr]= getattr(TriggerDef,attr)
				self.by_value[getattr(TriggerDef,attr)]= attr

# sample code
if __name__=='__main__':
	tdef= TriggerDef()

	# accessing a trigger value as a member variable
	print( 'INIT =', tdef.INIT )

	# check whether the trigger name is defined
	print( '\nINIT in tdef.by_key?')
	print( 'INIT' in tdef.by_key )

	# check whether the trigger value is defined
	print( '\n255 in tdef.by_value?' )
	print( 255 in tdef.by_value )

	# print all trigger names and associated values
	print( '\ntdef.by_key' )
	print( tdef.by_key )

	# print all trigger values and associated names
	print( '\ntdef.by_value' )
	print( tdef.by_value )
