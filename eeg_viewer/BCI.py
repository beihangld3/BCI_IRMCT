import pylibtobiic, pylibtobiid, pylibtobicore, pytpstreamer
import atexit
from socket import *

class BciInterface:
    def __init__(self):
	# Setup TOBI interfaces iC and iD
	# set up IC objects
        #self.ic_msg = pylibtobiic.ICMessage()
        #self.ic_serializer = pylibtobiic.ICSerializer(self.ic_msg, True)

        # set up ID objects
        self.id_msg_bus = pylibtobiid.IDMessage()
	self.id_msg_bus.SetBlockIdx(100)
	self.id_msg_bus.SetDescription("EEG_scope")
	self.id_msg_bus.SetFamilyType(0)
        self.id_msg_dev = pylibtobiid.IDMessage()
	self.id_msg_dev.SetBlockIdx(100)
	self.id_msg_dev.SetDescription("EEG_scope")
	self.id_msg_dev.SetFamilyType(0)

        self.id_serializer_bus = pylibtobiid.IDSerializer(self.id_msg_bus, True)
	self.id_serializer_dev = pylibtobiid.IDSerializer(self.id_msg_dev, True)
	
	# Bind sockets for iC and iD, hardcoded thanks to the new loop. 
	# I could retrieve it from the nameserver in the future
	#self.iCIP = '127.0.0.1'
	#self.iCport = 9503
	
	self.iDIP_bus = '127.0.0.1'
	self.iDport_bus = 8126
	self.iDIP_dev = '127.0.0.1'
	self.iDport_dev = 8127
	
	#self.icStreamer = pytpstreamer.TPStreamer()	
	self.idStreamer_bus = pytpstreamer.TPStreamer()
	self.idStreamer_dev = pytpstreamer.TPStreamer()
	self.iDsock_bus = socket(AF_INET, SOCK_STREAM)
        self.iDsock_bus.connect((self.iDIP_bus, self.iDport_bus))
	self.iDsock_dev = socket(AF_INET, SOCK_STREAM)
        self.iDsock_dev.connect((self.iDIP_dev, self.iDport_dev))
        #self.iCsock = socket(AF_INET, SOCK_STREAM)
        #self.iCsock.setsockopt(SOL_SOCKET, SO_REUSEADDR, 1)
        #self.iCsock.bind((self.iCIP, self.iCport))
	#self.iCsock.listen(1)
	#self.conn, self.address = self.iCsock.accept()

        #print 'Protocol is listening for iC data on ip %s, port %d' % (self.iCIP, self.iCport)
        #self.iCsock.setblocking(0)
        #atexit.register(self.iCsock.close) # close socket on program termination, no matter what!

        # print 'Protocol is listening for iD event data on ip %s, port %d' % (self.iDIP_bus, self.iDport_bus)
        # print 'Protocol is listening for iD command data on ip %s, port %d' % (self.iDIP_dev, self.iDport_dev)
        self.iDsock_bus.setblocking(0)
        self.iDsock_dev.setblocking(0)
        atexit.register(self.iDsock_bus.close) # close socket on program termination, no matter what!
        atexit.register(self.iDsock_dev.close) # close socket on program termination, no matter what!

